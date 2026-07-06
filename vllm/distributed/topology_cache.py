# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch


GroupT = TypeVar("GroupT")
GroupBuilder = Callable[[str, list[list[int]]], GroupT]


@dataclass(frozen=True)
class TopologyDescriptor:
    world_size: int
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    prefill_context_parallel_size: int = 1
    decode_context_parallel_size: int = 1
    data_parallel_size: int = 1

    def __post_init__(self) -> None:
        fields = {
            "world_size": self.world_size,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "prefill_context_parallel_size": self.prefill_context_parallel_size,
            "decode_context_parallel_size": self.decode_context_parallel_size,
            "data_parallel_size": self.data_parallel_size,
        }
        for name, value in fields.items():
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")

        expected_world_size = (
            self.data_parallel_size
            * self.pipeline_parallel_size
            * self.prefill_context_parallel_size
            * self.tensor_parallel_size
        )
        if self.world_size != expected_world_size:
            raise ValueError(
                "world_size must equal data_parallel_size * "
                "pipeline_parallel_size * prefill_context_parallel_size * "
                "tensor_parallel_size; got "
                f"{self.world_size=} and {expected_world_size=}"
            )

        if self.decode_context_parallel_size > self.tensor_parallel_size:
            raise ValueError(
                "decode_context_parallel_size must be <= tensor_parallel_size; "
                f"got {self.decode_context_parallel_size=} and "
                f"{self.tensor_parallel_size=}"
            )
        if self.tensor_parallel_size % self.decode_context_parallel_size != 0:
            raise ValueError(
                "decode_context_parallel_size must divide tensor_parallel_size; "
                f"got {self.decode_context_parallel_size=} and "
                f"{self.tensor_parallel_size=}"
            )

    @property
    def key(self) -> tuple[int, int, int, int, int, int]:
        return (
            self.world_size,
            self.data_parallel_size,
            self.pipeline_parallel_size,
            self.prefill_context_parallel_size,
            self.decode_context_parallel_size,
            self.tensor_parallel_size,
        )


@dataclass(frozen=True)
class TopologyGroupLayout:
    tp: list[list[int]]
    dcp: list[list[int]]
    pcp: list[list[int]]
    pp: list[list[int]]
    dp: list[list[int]]

    def groups(self) -> tuple[tuple[str, list[list[int]]], ...]:
        return (
            ("tp", self.tp),
            ("dcp", self.dcp),
            ("pcp", self.pcp),
            ("pp", self.pp),
            ("dp", self.dp),
        )


def _tolist(group_ranks: tuple[torch.Tensor, ...]) -> list[list[int]]:
    return [group.tolist() for group in group_ranks]


def plan_topology_groups(descriptor: TopologyDescriptor) -> TopologyGroupLayout:
    """Plan model-parallel rank groups using vLLM's existing layout order."""
    all_ranks = torch.arange(descriptor.world_size).reshape(
        -1,
        descriptor.data_parallel_size,
        descriptor.pipeline_parallel_size,
        descriptor.prefill_context_parallel_size,
        descriptor.tensor_parallel_size,
    )

    tp = _tolist(
        all_ranks.view(-1, descriptor.tensor_parallel_size).unbind(0)
    )
    dcp = _tolist(
        all_ranks.reshape(-1, descriptor.decode_context_parallel_size).unbind(0)
    )
    pcp = _tolist(
        all_ranks.transpose(3, 4)
        .reshape(-1, descriptor.prefill_context_parallel_size)
        .unbind(0)
    )
    pp = _tolist(
        all_ranks.transpose(2, 4)
        .reshape(-1, descriptor.pipeline_parallel_size)
        .unbind(0)
    )
    dp = _tolist(
        all_ranks.transpose(1, 4)
        .reshape(-1, descriptor.data_parallel_size)
        .unbind(0)
    )

    return TopologyGroupLayout(tp=tp, dcp=dcp, pcp=pcp, pp=pp, dp=dp)


def parse_topology_descriptors(
    spec: str,
    *,
    world_size: int,
    data_parallel_size: int = 1,
    prefill_context_parallel_size: int = 1,
    decode_context_parallel_size: int = 1,
) -> list[TopologyDescriptor]:
    if not spec.strip():
        return []

    aliases = {
        "tp": "tensor_parallel_size",
        "tensor_parallel_size": "tensor_parallel_size",
        "pp": "pipeline_parallel_size",
        "pipeline_parallel_size": "pipeline_parallel_size",
        "pcp": "prefill_context_parallel_size",
        "prefill_context_parallel_size": "prefill_context_parallel_size",
        "dcp": "decode_context_parallel_size",
        "decode_context_parallel_size": "decode_context_parallel_size",
        "dp": "data_parallel_size",
        "data_parallel_size": "data_parallel_size",
    }
    descriptors: list[TopologyDescriptor] = []
    for entry in spec.split(";"):
        entry = entry.strip()
        if not entry:
            continue

        values = {
            "world_size": world_size,
            "data_parallel_size": data_parallel_size,
            "prefill_context_parallel_size": prefill_context_parallel_size,
            "decode_context_parallel_size": decode_context_parallel_size,
        }
        for item in entry.split(","):
            if "=" not in item:
                raise ValueError(f"invalid topology field {item!r}")
            raw_key, raw_value = item.split("=", 1)
            key = raw_key.strip()
            if key not in aliases:
                raise ValueError(f"unknown topology field {key!r}")
            try:
                value = int(raw_value.strip())
            except ValueError as e:
                raise ValueError(
                    f"topology field {key!r} must be an integer"
                ) from e
            values[aliases[key]] = value

        descriptors.append(TopologyDescriptor(**values))

    return descriptors


@dataclass
class TopologyGroupSnapshot(Generic[GroupT]):
    descriptor: TopologyDescriptor
    layout: TopologyGroupLayout
    tp: GroupT
    dcp: GroupT
    pcp: GroupT
    pp: GroupT
    dp: GroupT
    destroyed: bool = False

    def groups(self) -> tuple[tuple[str, GroupT], ...]:
        return (
            ("tp", self.tp),
            ("dcp", self.dcp),
            ("pcp", self.pcp),
            ("pp", self.pp),
            ("dp", self.dp),
        )

    def destroy(self) -> None:
        if self.destroyed:
            return
        for _, group in self.groups():
            destroy = getattr(group, "destroy", None)
            if destroy is not None:
                destroy()
        self.destroyed = True


class TopologyStateCache(Generic[GroupT]):

    def __init__(self, group_builder: GroupBuilder[GroupT]) -> None:
        self._group_builder = group_builder
        self._snapshots: dict[
            tuple[int, int, int, int, int, int], TopologyGroupSnapshot[GroupT]
        ] = {}
        self._active_key: tuple[int, int, int, int, int, int] | None = None

    def prebuild(self, descriptor: TopologyDescriptor) -> TopologyGroupSnapshot[GroupT]:
        key = descriptor.key
        snapshot = self._snapshots.get(key)
        if snapshot is not None:
            return snapshot

        layout = plan_topology_groups(descriptor)
        groups = {
            group_name: self._group_builder(group_name, group_ranks)
            for group_name, group_ranks in layout.groups()
        }
        snapshot = TopologyGroupSnapshot(
            descriptor=descriptor,
            layout=layout,
            tp=groups["tp"],
            dcp=groups["dcp"],
            pcp=groups["pcp"],
            pp=groups["pp"],
            dp=groups["dp"],
        )
        self._snapshots[key] = snapshot
        return snapshot

    def activate(
        self, descriptor: TopologyDescriptor
    ) -> TopologyGroupSnapshot[GroupT]:
        snapshot = self._snapshots[descriptor.key]
        self._active_key = descriptor.key
        return snapshot

    def destroy(self) -> None:
        for snapshot in self._snapshots.values():
            snapshot.destroy()
        self._snapshots.clear()
        self._active_key = None
