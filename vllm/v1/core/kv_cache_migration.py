# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for runtime KV cache relayout.

This module builds the layer/head/capacity plan for runtime KV migration and
provides a standard attention KV tensor copy core for future engine integration.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
import copy
from dataclasses import dataclass, replace
from enum import Enum

import torch

from vllm.distributed.topology_cache import (
    TopologyDescriptor,
    plan_topology_groups,
)
from vllm.distributed.utils import get_pp_indices
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
)


class RuntimeKVMigrationPolicy(str, Enum):
    MIGRATE = "migrate"
    RECOMPUTE = "recompute"


@dataclass(frozen=True)
class RuntimeKVLayerPartition:
    pp_rank: int
    layer_indices: range


@dataclass(frozen=True)
class RuntimeKVHeadPartition:
    tp_rank: int
    head_indices: range


@dataclass(frozen=True)
class RuntimeKVMigrationPlan:
    source_topology: TopologyDescriptor
    target_topology: TopologyDescriptor
    policy: RuntimeKVMigrationPolicy
    reason: str
    pp_partitions: list[RuntimeKVLayerPartition]
    tp_partitions: list[RuntimeKVHeadPartition]
    live_blocks: int
    target_num_blocks: int
    layer_names: tuple[str, ...] = ()
    global_num_kv_heads: int = 0


@dataclass(frozen=True)
class RuntimeKVMigrationStep:
    pp_rank: int
    tp_rank: int
    layer_indices: range
    head_indices: range
    block_ids: tuple[int, ...]


@dataclass(frozen=True)
class RuntimeKVMigrationCopyStats:
    migration_steps: int
    tensor_copies: int


@dataclass(frozen=True)
class RuntimeKVSourceTensor:
    tensor: torch.Tensor
    head_indices: tuple[int, ...]


@dataclass(frozen=True)
class RuntimeKVBlockMappingPlan:
    block_mapping: dict[int, int]
    live_blocks: int


_LAYER_INDEX_PATTERN = re.compile(r"(?:^|\.)(?:layers|h|blocks)\.(\d+)(?:\.|$)")


def get_runtime_topology_kv_rank(
    descriptor: TopologyDescriptor,
    *,
    rank: int,
) -> tuple[int, int]:
    """Return ``(pp_rank, tp_rank)`` for a global rank in a topology."""
    if descriptor.data_parallel_size != 1:
        raise ValueError("runtime KV migration currently requires DP size 1")
    if descriptor.prefill_context_parallel_size != 1:
        raise ValueError("runtime KV migration currently requires PCP size 1")
    if rank < 0 or rank >= descriptor.world_size:
        raise ValueError(
            f"rank must be in [0, {descriptor.world_size}), got {rank}"
        )

    layout = plan_topology_groups(descriptor)
    tp_rank = next(
        (
            group.index(rank)
            for group in layout.tp
            if rank in group
        ),
        None,
    )
    pp_rank = next(
        (
            group.index(rank)
            for group in layout.pp
            if rank in group
        ),
        None,
    )
    if tp_rank is None or pp_rank is None:
        raise ValueError(f"rank {rank} is not covered by topology layout")
    return pp_rank, tp_rank


def _layer_index(layer_name: str) -> int:
    match = _LAYER_INDEX_PATTERN.search(layer_name)
    if match is None:
        raise ValueError(f"cannot infer layer index from {layer_name!r}")
    return int(match.group(1))


def _collect_layer_names_by_index(kv_cache_config: KVCacheConfig) -> tuple[str, ...]:
    layer_names_by_index: dict[int, str] = {}
    for group in kv_cache_config.kv_cache_groups:
        for layer_name in group.layer_names:
            layer_index = _layer_index(layer_name)
            if layer_index in layer_names_by_index:
                raise ValueError(
                    "KV cache migration requires unique layer indices; "
                    f"layer index {layer_index} appears in both "
                    f"{layer_names_by_index[layer_index]!r} and {layer_name!r}"
                )
            layer_names_by_index[layer_index] = layer_name

    expected = list(range(len(layer_names_by_index)))
    actual = sorted(layer_names_by_index)
    if actual != expected:
        raise ValueError(
            "KV cache migration requires dense layer indices; "
            f"got {actual}, expected {expected}"
        )
    return tuple(layer_names_by_index[layer_index] for layer_index in expected)


def _get_num_kv_heads(kv_cache_config: KVCacheConfig) -> int:
    num_kv_heads: set[int] = set()
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        if isinstance(spec, AttentionSpec):
            num_kv_heads.add(spec.num_kv_heads)
    if not num_kv_heads:
        raise ValueError("KV cache migration requires attention KV cache groups")
    if len(num_kv_heads) != 1:
        raise ValueError(
            "KV cache migration requires a single KV head count across groups"
        )
    return num_kv_heads.pop()


def _with_global_kv_heads(
    spec: KVCacheSpec,
    *,
    target_tp_size: int,
) -> KVCacheSpec:
    spec = copy.deepcopy(spec)
    if isinstance(spec, AttentionSpec):
        return replace(
            spec,
            num_kv_heads=spec.num_kv_heads * target_tp_size,
        )
    return spec


def build_global_runtime_kv_cache_config(
    worker_kv_cache_configs: Sequence[KVCacheConfig],
    *,
    target_topology: TopologyDescriptor,
) -> KVCacheConfig:
    """Merge worker-local KV config into a global planning view.

    Scheduler KV config intentionally uses an arbitrary worker's layer names.
    Runtime KV migration needs the union of PP-local layer names and global KV
    head count, but it must not change scheduler allocation semantics.
    """
    if not worker_kv_cache_configs:
        raise ValueError("runtime KV migration requires worker KV cache configs")

    first_config = worker_kv_cache_configs[0]
    for config in worker_kv_cache_configs:
        if config.num_blocks != first_config.num_blocks:
            raise ValueError(
                "runtime KV migration requires equal KV block counts across "
                "workers"
            )
        if len(config.kv_cache_groups) != len(first_config.kv_cache_groups):
            raise ValueError(
                "runtime KV migration requires equal KV cache group counts "
                "across workers"
            )

    global_groups: list[KVCacheGroupSpec] = []
    for group_index in range(len(first_config.kv_cache_groups)):
        layer_names: list[str] = []
        seen_layer_names: set[str] = set()
        group_spec: KVCacheSpec | None = None
        is_eagle_group = False
        for config in worker_kv_cache_configs:
            group = config.kv_cache_groups[group_index]
            is_eagle_group = is_eagle_group or group.is_eagle_group
            if group.layer_names and group_spec is None:
                group_spec = group.kv_cache_spec
            for layer_name in group.layer_names:
                if layer_name in seen_layer_names:
                    continue
                seen_layer_names.add(layer_name)
                layer_names.append(layer_name)

        if group_spec is None:
            group_spec = first_config.kv_cache_groups[group_index].kv_cache_spec
        layer_names.sort(key=lambda name: (_layer_index(name), name))
        global_groups.append(
            KVCacheGroupSpec(
                layer_names=layer_names,
                kv_cache_spec=_with_global_kv_heads(
                    group_spec,
                    target_tp_size=target_topology.tensor_parallel_size,
                ),
                is_eagle_group=is_eagle_group,
            )
        )

    return KVCacheConfig(
        num_blocks=first_config.num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=global_groups,
    )


def _build_pp_partitions(
    *,
    num_layers: int,
    target_pp_size: int,
) -> list[RuntimeKVLayerPartition]:
    partitions = [
        RuntimeKVLayerPartition(
            pp_rank=pp_rank,
            layer_indices=range(*get_pp_indices(num_layers, pp_rank, target_pp_size)),
        )
        for pp_rank in range(target_pp_size)
    ]

    covered = [
        layer_index
        for partition in partitions
        for layer_index in partition.layer_indices
    ]
    expected = list(range(num_layers))
    if sorted(covered) != expected or len(set(covered)) != len(covered):
        raise ValueError(
            "target PP layer partitions must cover each layer exactly once"
        )
    return partitions


def _build_tp_partitions(
    *,
    num_kv_heads: int,
    target_tp_size: int,
) -> list[RuntimeKVHeadPartition]:
    if num_kv_heads % target_tp_size != 0:
        raise ValueError(
            "KV heads must be divisible by target tensor parallel size; "
            f"got {num_kv_heads=} and {target_tp_size=}"
        )
    heads_per_rank = num_kv_heads // target_tp_size
    partitions = [
        RuntimeKVHeadPartition(
            tp_rank=tp_rank,
            head_indices=range(
                tp_rank * heads_per_rank,
                (tp_rank + 1) * heads_per_rank,
            ),
        )
        for tp_rank in range(target_tp_size)
    ]

    covered = [
        head_index
        for partition in partitions
        for head_index in partition.head_indices
    ]
    expected = list(range(num_kv_heads))
    if sorted(covered) != expected or len(set(covered)) != len(covered):
        raise ValueError(
            "target TP KV head partitions must cover each head exactly once"
        )
    return partitions


def build_runtime_kv_migration_plan(
    *,
    source_topology: TopologyDescriptor,
    target_topology: TopologyDescriptor,
    kv_cache_config: KVCacheConfig,
    live_blocks: int,
) -> RuntimeKVMigrationPlan:
    if live_blocks < 0:
        raise ValueError("live_blocks must be >= 0")

    layer_names = _collect_layer_names_by_index(kv_cache_config)
    num_kv_heads = _get_num_kv_heads(kv_cache_config)
    pp_partitions = _build_pp_partitions(
        num_layers=len(layer_names),
        target_pp_size=target_topology.pipeline_parallel_size,
    )
    tp_partitions = _build_tp_partitions(
        num_kv_heads=num_kv_heads,
        target_tp_size=target_topology.tensor_parallel_size,
    )

    if live_blocks > kv_cache_config.num_blocks:
        policy = RuntimeKVMigrationPolicy.RECOMPUTE
        reason = "insufficient_target_kv_capacity"
    else:
        policy = RuntimeKVMigrationPolicy.MIGRATE
        reason = "capacity_available"

    return RuntimeKVMigrationPlan(
        source_topology=source_topology,
        target_topology=target_topology,
        policy=policy,
        reason=reason,
        pp_partitions=pp_partitions,
        tp_partitions=tp_partitions,
        live_blocks=live_blocks,
        target_num_blocks=kv_cache_config.num_blocks,
        layer_names=layer_names,
        global_num_kv_heads=num_kv_heads,
    )


def iter_layerwise_kv_migration_steps(
    plan: RuntimeKVMigrationPlan,
    *,
    block_ids: Sequence[int],
    max_blocks_per_step: int,
) -> Iterator[RuntimeKVMigrationStep]:
    if max_blocks_per_step < 1:
        raise ValueError("max_blocks_per_step must be >= 1")
    if plan.policy != RuntimeKVMigrationPolicy.MIGRATE:
        return
    if len(block_ids) != plan.live_blocks:
        raise ValueError(
            "block_ids length must match plan.live_blocks; "
            f"got {len(block_ids)=} and {plan.live_blocks=}"
        )

    block_id_tuple = tuple(block_ids)
    for pp_partition in plan.pp_partitions:
        for tp_partition in plan.tp_partitions:
            for start in range(0, len(block_id_tuple), max_blocks_per_step):
                yield RuntimeKVMigrationStep(
                    pp_rank=pp_partition.pp_rank,
                    tp_rank=tp_partition.tp_rank,
                    layer_indices=pp_partition.layer_indices,
                    head_indices=tp_partition.head_indices,
                    block_ids=block_id_tuple[start : start + max_blocks_per_step],
                )


def build_runtime_kv_block_mapping(
    *,
    request_block_ids: Mapping[str, Sequence[int]],
    target_num_blocks: int,
) -> RuntimeKVBlockMappingPlan:
    if target_num_blocks < 0:
        raise ValueError("target_num_blocks must be >= 0")

    source_block_ids: list[int] = []
    seen_source_blocks: set[int] = set()
    for block_ids in request_block_ids.values():
        for block_id in block_ids:
            if block_id < 0:
                raise ValueError(f"source block id must be >= 0, got {block_id}")
            if block_id == 0:
                continue
            if block_id in seen_source_blocks:
                continue
            seen_source_blocks.add(block_id)
            source_block_ids.append(block_id)

    usable_target_blocks = max(target_num_blocks - 1, 0)
    if len(source_block_ids) > usable_target_blocks:
        raise ValueError(
            "target KV capacity is insufficient for live blocks; "
            f"got {len(source_block_ids)=} and {target_num_blocks=}"
        )

    if all(0 < block_id < target_num_blocks for block_id in source_block_ids):
        block_mapping = {
            source_block_id: source_block_id
            for source_block_id in source_block_ids
        }
    else:
        block_mapping = {
            source_block_id: target_block_id
            for target_block_id, source_block_id in enumerate(
                source_block_ids,
                start=1,
            )
        }

    return RuntimeKVBlockMappingPlan(
        block_mapping=block_mapping,
        live_blocks=len(source_block_ids),
    )


def _partition_for_layer(
    partitions: Sequence[RuntimeKVLayerPartition], layer_index: int
) -> RuntimeKVLayerPartition:
    for partition in partitions:
        if layer_index in partition.layer_indices:
            return partition
    raise ValueError(f"layer {layer_index} is not covered by source PP partitions")


def _partition_for_head(
    partitions: Sequence[RuntimeKVHeadPartition], head_index: int
) -> RuntimeKVHeadPartition:
    for partition in partitions:
        if head_index in partition.head_indices:
            return partition
    raise ValueError(f"KV head {head_index} is not covered by source TP partitions")


def _validate_block_mapping(
    *,
    block_mapping: Mapping[int, int],
    live_blocks: int,
    target_num_blocks: int,
) -> tuple[int, ...]:
    if len(block_mapping) != live_blocks:
        raise ValueError(
            "block_mapping length must match plan.live_blocks; "
            f"got {len(block_mapping)=} and {live_blocks=}"
        )
    for source_block_id, target_block_id in block_mapping.items():
        if source_block_id < 0:
            raise ValueError(f"source block id must be >= 0, got {source_block_id}")
        if target_block_id < 0:
            raise ValueError(f"target block id must be >= 0, got {target_block_id}")
        if target_block_id == 0:
            raise ValueError("target block id must exclude the null block 0")
        if target_block_id >= target_num_blocks:
            raise ValueError(
                "target block id must be smaller than target_num_blocks; "
                f"got {target_block_id=} and {target_num_blocks=}"
            )
    if len(set(block_mapping.values())) != len(block_mapping):
        raise ValueError("block_mapping contains duplicate target block ids")
    return tuple(block_mapping.keys())


def _validate_attention_kv_tensor(
    tensor: torch.Tensor,
    *,
    expected_local_heads: int,
    role: str,
    layer_name: str,
    pp_rank: int,
    tp_rank: int,
) -> None:
    if tensor.ndim != 5 or tensor.shape[1] != 2:
        raise ValueError(
            "runtime KV migration currently supports only the standard "
            "attention KV cache layout "
            "(num_blocks, 2, block_size, num_kv_heads, head_size); "
            f"{role} shard pp={pp_rank}, tp={tp_rank}, layer={layer_name!r} "
            f"has shape {tuple(tensor.shape)}"
        )
    if tensor.shape[3] != expected_local_heads:
        raise ValueError(
            f"{role} shard pp={pp_rank}, tp={tp_rank}, layer={layer_name!r} "
            f"has {tensor.shape[3]} local KV heads, expected "
            f"{expected_local_heads}"
        )


def _get_layer_cache(
    kv_caches: Mapping[
        tuple[int, int],
        Mapping[str, torch.Tensor | RuntimeKVSourceTensor],
    ],
    *,
    pp_rank: int,
    tp_rank: int,
    layer_name: str,
    role: str,
) -> torch.Tensor | RuntimeKVSourceTensor:
    shard_key = (pp_rank, tp_rank)
    try:
        shard = kv_caches[shard_key]
    except KeyError as e:
        raise ValueError(
            f"missing {role} KV cache shard for pp={pp_rank}, tp={tp_rank}"
        ) from e
    try:
        return shard[layer_name]
    except KeyError as e:
        raise ValueError(
            f"missing {role} KV cache layer {layer_name!r} on "
            f"pp={pp_rank}, tp={tp_rank}"
        ) from e


def _get_local_layer_cache(
    kv_caches: Mapping[str, torch.Tensor],
    *,
    layer_name: str,
    role: str,
    pp_rank: int,
    tp_rank: int,
) -> torch.Tensor:
    try:
        return kv_caches[layer_name]
    except KeyError as e:
        raise ValueError(
            f"missing {role} KV cache layer {layer_name!r} on "
            f"pp={pp_rank}, tp={tp_rank}"
        ) from e


def _require_target_pp_partition(
    partitions: Sequence[RuntimeKVLayerPartition],
    target_pp_rank: int,
) -> RuntimeKVLayerPartition:
    for partition in partitions:
        if partition.pp_rank == target_pp_rank:
            return partition
    raise ValueError(f"target PP rank {target_pp_rank} is not in the migration plan")


def _require_target_tp_partition(
    partitions: Sequence[RuntimeKVHeadPartition],
    target_tp_rank: int,
) -> RuntimeKVHeadPartition:
    for partition in partitions:
        if partition.tp_rank == target_tp_rank:
            return partition
    raise ValueError(f"target TP rank {target_tp_rank} is not in the migration plan")


def migrate_runtime_kv_cache_shard(
    plan: RuntimeKVMigrationPlan,
    *,
    source_kv_caches: Mapping[
        tuple[int, int],
        Mapping[str, torch.Tensor | RuntimeKVSourceTensor],
    ],
    target_kv_caches: Mapping[str, torch.Tensor],
    target_pp_rank: int,
    target_tp_rank: int,
    block_mapping: Mapping[int, int],
    max_blocks_per_step: int,
) -> RuntimeKVMigrationCopyStats:
    """Copy KV cache data for a single target ``(pp_rank, tp_rank)`` shard."""
    if plan.policy != RuntimeKVMigrationPolicy.MIGRATE:
        raise ValueError(
            "runtime KV cache migration cannot execute a recompute plan; "
            f"got policy={plan.policy.value!r} and reason={plan.reason!r}"
        )
    if not plan.layer_names:
        raise ValueError("runtime KV cache migration plan is missing layer names")

    _require_target_pp_partition(plan.pp_partitions, target_pp_rank)
    target_tp_partition = _require_target_tp_partition(
        plan.tp_partitions, target_tp_rank
    )

    source_block_ids = _validate_block_mapping(
        block_mapping=block_mapping,
        live_blocks=plan.live_blocks,
        target_num_blocks=plan.target_num_blocks,
    )
    source_pp_partitions = _build_pp_partitions(
        num_layers=len(plan.layer_names),
        target_pp_size=plan.source_topology.pipeline_parallel_size,
    )
    source_tp_partitions = _build_tp_partitions(
        num_kv_heads=plan.global_num_kv_heads
        or sum(len(partition.head_indices) for partition in plan.tp_partitions),
        target_tp_size=plan.source_topology.tensor_parallel_size,
    )

    migration_steps = 0
    tensor_copies = 0
    target_local_heads = len(target_tp_partition.head_indices)
    for step in iter_layerwise_kv_migration_steps(
        plan,
        block_ids=source_block_ids,
        max_blocks_per_step=max_blocks_per_step,
    ):
        if step.pp_rank != target_pp_rank or step.tp_rank != target_tp_rank:
            continue

        migration_steps += 1
        for layer_index in step.layer_indices:
            layer_name = plan.layer_names[layer_index]
            source_pp_partition = _partition_for_layer(
                source_pp_partitions, layer_index
            )
            target_tensor = _get_local_layer_cache(
                target_kv_caches,
                layer_name=layer_name,
                role="target",
                pp_rank=target_pp_rank,
                tp_rank=target_tp_rank,
            )
            _validate_attention_kv_tensor(
                target_tensor,
                expected_local_heads=target_local_heads,
                role="target",
                layer_name=layer_name,
                pp_rank=target_pp_rank,
                tp_rank=target_tp_rank,
            )

            for global_head_index in step.head_indices:
                source_tp_partition = _partition_for_head(
                    source_tp_partitions, global_head_index
                )
                source_cache = _get_layer_cache(
                    source_kv_caches,
                    pp_rank=source_pp_partition.pp_rank,
                    tp_rank=source_tp_partition.tp_rank,
                    layer_name=layer_name,
                    role="source",
                )
                if isinstance(source_cache, RuntimeKVSourceTensor):
                    source_tensor = source_cache.tensor
                    if global_head_index not in source_cache.head_indices:
                        raise ValueError(
                            "runtime KV source shard for "
                            f"pp={source_pp_partition.pp_rank}, "
                            f"tp={source_tp_partition.tp_rank}, "
                            f"layer={layer_name!r} does not contain "
                            f"KV head {global_head_index}"
                        )
                    if not all(
                        head_index in source_tp_partition.head_indices
                        for head_index in source_cache.head_indices
                    ):
                        raise ValueError(
                            "runtime KV source tensor head metadata is "
                            "outside its source TP partition"
                        )
                    _validate_attention_kv_tensor(
                        source_tensor,
                        expected_local_heads=len(source_cache.head_indices),
                        role="source",
                        layer_name=layer_name,
                        pp_rank=source_pp_partition.pp_rank,
                        tp_rank=source_tp_partition.tp_rank,
                    )
                    source_local_head = source_cache.head_indices.index(
                        global_head_index
                    )
                else:
                    source_tensor = source_cache
                    _validate_attention_kv_tensor(
                        source_tensor,
                        expected_local_heads=len(source_tp_partition.head_indices),
                        role="source",
                        layer_name=layer_name,
                        pp_rank=source_pp_partition.pp_rank,
                        tp_rank=source_tp_partition.tp_rank,
                    )
                    source_local_head = (
                        global_head_index - source_tp_partition.head_indices.start
                    )
                target_local_head = (
                    global_head_index - target_tp_partition.head_indices.start
                )
                for source_block_id in step.block_ids:
                    target_block_id = block_mapping[source_block_id]
                    if source_block_id >= source_tensor.shape[0]:
                        raise ValueError(
                            f"source block id {source_block_id} exceeds "
                            f"source shard capacity {source_tensor.shape[0]}"
                        )
                    if target_block_id >= target_tensor.shape[0]:
                        raise ValueError(
                            f"target block id {target_block_id} exceeds "
                            f"target shard capacity {target_tensor.shape[0]}"
                        )
                    target_tensor[
                        target_block_id,
                        :,
                        :,
                        target_local_head,
                        :,
                    ].copy_(
                        source_tensor[
                            source_block_id,
                            :,
                            :,
                            source_local_head,
                            :,
                        ],
                        non_blocking=True,
                    )
                    tensor_copies += 1

    return RuntimeKVMigrationCopyStats(
        migration_steps=migration_steps,
        tensor_copies=tensor_copies,
    )


def migrate_runtime_kv_cache(
    plan: RuntimeKVMigrationPlan,
    *,
    source_kv_caches: Mapping[
        tuple[int, int],
        Mapping[str, torch.Tensor | RuntimeKVSourceTensor],
    ],
    target_kv_caches: Mapping[tuple[int, int], Mapping[str, torch.Tensor]],
    block_mapping: Mapping[int, int],
    max_blocks_per_step: int,
) -> RuntimeKVMigrationCopyStats:
    """Copy standard attention KV cache tensors according to a migration plan.

    The input cache dictionaries are keyed by logical ``(pp_rank, tp_rank)``
    pairs. Each layer tensor must use the standard visible attention layout
    ``(num_blocks, 2, block_size, local_num_kv_heads, head_size)``.
    """
    if plan.policy != RuntimeKVMigrationPolicy.MIGRATE:
        raise ValueError(
            "runtime KV cache migration cannot execute a recompute plan; "
            f"got policy={plan.policy.value!r} and reason={plan.reason!r}"
        )

    migration_steps = 0
    tensor_copies = 0
    for pp_partition in plan.pp_partitions:
        for tp_partition in plan.tp_partitions:
            target_shard = target_kv_caches[
                (pp_partition.pp_rank, tp_partition.tp_rank)
            ]
            stats = migrate_runtime_kv_cache_shard(
                plan,
                source_kv_caches=source_kv_caches,
                target_kv_caches=target_shard,
                target_pp_rank=pp_partition.pp_rank,
                target_tp_rank=tp_partition.tp_rank,
                block_mapping=block_mapping,
                max_blocks_per_step=max_blocks_per_step,
            )
            migration_steps += stats.migration_steps
            tensor_copies += stats.tensor_copies

    return RuntimeKVMigrationCopyStats(
        migration_steps=migration_steps,
        tensor_copies=tensor_copies,
    )
