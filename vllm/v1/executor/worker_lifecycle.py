# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker active/standby state helpers for runtime topology changes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from vllm.v1.worker.lifecycle import WorkerRole


@dataclass(frozen=True)
class WorkerActivationPlan:
    max_world_size: int
    active_ranks: frozenset[int]

    @property
    def standby_ranks(self) -> frozenset[int]:
        return frozenset(range(self.max_world_size)) - self.active_ranks

    @classmethod
    def from_world_size(
        cls,
        max_world_size: int,
        active_world_size: int,
    ) -> "WorkerActivationPlan":
        if max_world_size <= 0:
            raise ValueError("max_world_size must be positive")
        if active_world_size <= 0:
            raise ValueError("active_world_size must be positive")
        if active_world_size > max_world_size:
            raise ValueError(
                "active_world_size cannot exceed max_world_size: "
                f"{active_world_size} > {max_world_size}"
            )
        return cls(
            max_world_size=max_world_size,
            active_ranks=frozenset(range(active_world_size)),
        )


class WorkerLifecyclePool:
    """Tracks which executor workers are active or standby."""

    def __init__(self, workers_by_rank: dict[int, Any]) -> None:
        if not workers_by_rank:
            raise ValueError("WorkerLifecyclePool requires at least one worker")
        self._workers_by_rank = dict(workers_by_rank)
        self._roles = {rank: WorkerRole.ACTIVE for rank in self._workers_by_rank}

    @classmethod
    def from_workers(cls, workers: Sequence[Any]) -> "WorkerLifecyclePool":
        workers_by_rank: dict[int, Any] = {}
        for index, worker in enumerate(workers):
            rank = int(getattr(worker, "rank", index))
            if rank in workers_by_rank:
                raise ValueError(f"Duplicate worker rank: {rank}")
            workers_by_rank[rank] = worker
        return cls(workers_by_rank)

    @property
    def active_ranks(self) -> tuple[int, ...]:
        return tuple(
            rank
            for rank, role in sorted(self._roles.items())
            if role is WorkerRole.ACTIVE
        )

    @property
    def standby_ranks(self) -> tuple[int, ...]:
        return tuple(
            rank
            for rank, role in sorted(self._roles.items())
            if role is WorkerRole.STANDBY
        )

    def role_for_rank(self, rank: int) -> WorkerRole:
        return self._roles[rank]

    def worker_for_rank(self, rank: int) -> Any:
        return self._workers_by_rank[rank]

    def apply_plan(self, plan: WorkerActivationPlan) -> None:
        expected_ranks = set(range(plan.max_world_size))
        actual_ranks = set(self._workers_by_rank)
        if actual_ranks != expected_ranks:
            raise ValueError(
                "Activation plan rank set does not match worker pool: "
                f"plan={sorted(expected_ranks)} workers={sorted(actual_ranks)}"
            )
        self._roles = {
            rank: (
                WorkerRole.ACTIVE
                if rank in plan.active_ranks
                else WorkerRole.STANDBY
            )
            for rank in self._workers_by_rank
        }

    def call_worker(
        self,
        rank: int,
        method: str,
        *args: Any,
        allow_standby: bool = False,
        **kwargs: Any,
    ) -> Any:
        role = self.role_for_rank(rank)
        if role is WorkerRole.STANDBY and not allow_standby:
            raise RuntimeError(
                f"Worker rank {rank} is standby and cannot execute {method}"
            )
        return getattr(self.worker_for_rank(rank), method)(*args, **kwargs)
