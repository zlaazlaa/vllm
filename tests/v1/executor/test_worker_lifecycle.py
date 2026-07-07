# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing as mp
import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from vllm.v1.executor.worker_lifecycle import (
    WorkerActivationPlan,
    WorkerLifecyclePool,
)
from vllm.v1.worker.lifecycle import WorkerRole
from vllm.v1.worker.worker_base import WorkerWrapperBase
from vllm.utils.network_utils import get_open_port


class FakeWorker:
    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.calls: list[str] = []

    def execute_model(self, *args, **kwargs):
        self.calls.append("execute_model")
        return f"rank-{self.rank}"

    def check_health(self) -> None:
        self.calls.append("check_health")


class FakeDistributedWorker:
    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.collective_calls = 0

    def execute_model(self, *_args, **_kwargs) -> int:
        tensor = torch.tensor([self.rank], dtype=torch.int64)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        self.collective_calls += 1
        return int(tensor.item())


def _worker_wake_collective_process(
    rank: int,
    world_size: int,
    init_method: str,
    queue: mp.Queue,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        wrapper = WorkerWrapperBase(rpc_rank=rank, global_rank=rank)
        wrapper.worker = FakeDistributedWorker(rank)  # type: ignore[assignment]
        wrapper.mm_receiver_cache = None

        standby_role = wrapper.apply_worker_activation_plan((0,))
        standby_rejected = False
        if rank == 1:
            with pytest.raises(RuntimeError, match="standby"):
                wrapper.execute_model(object())
            standby_rejected = True

        dist.barrier()
        active_role = wrapper.apply_worker_activation_plan((0, 1))
        collective_result = wrapper.execute_model(object())
        dist.barrier()

        queue.put(
            {
                "rank": rank,
                "pid": os.getpid(),
                "standby_role": standby_role.value,
                "active_role": active_role.value,
                "standby_rejected": standby_rejected,
                "collective_result": collective_result,
                "collective_calls": wrapper.worker.collective_calls,
            }
        )
    finally:
        dist.destroy_process_group()


def test_activation_plan_uses_lowest_ranks_for_target_world_size():
    plan = WorkerActivationPlan.from_world_size(max_world_size=4, active_world_size=2)

    assert plan.active_ranks == frozenset({0, 1})
    assert plan.standby_ranks == frozenset({2, 3})


def test_activation_plan_rejects_invalid_world_size():
    with pytest.raises(ValueError, match="active_world_size"):
        WorkerActivationPlan.from_world_size(max_world_size=2, active_world_size=3)

    with pytest.raises(ValueError, match="positive"):
        WorkerActivationPlan.from_world_size(max_world_size=2, active_world_size=0)


def test_worker_lifecycle_pool_tracks_active_and_standby_roles():
    workers = [FakeWorker(rank) for rank in range(4)]
    pool = WorkerLifecyclePool.from_workers(workers)

    pool.apply_plan(WorkerActivationPlan.from_world_size(4, 2))

    assert pool.active_ranks == (0, 1)
    assert pool.standby_ranks == (2, 3)
    assert pool.role_for_rank(0) is WorkerRole.ACTIVE
    assert pool.role_for_rank(3) is WorkerRole.STANDBY


def test_worker_lifecycle_pool_rejects_execution_on_standby_worker():
    workers = [FakeWorker(rank) for rank in range(3)]
    pool = WorkerLifecyclePool.from_workers(workers)
    pool.apply_plan(WorkerActivationPlan.from_world_size(3, 1))

    with pytest.raises(RuntimeError, match="standby"):
        pool.call_worker(2, "execute_model", object())

    assert workers[2].calls == []


def test_worker_lifecycle_pool_wake_standby_without_replacing_worker():
    workers = [FakeWorker(rank) for rank in range(3)]
    pool = WorkerLifecyclePool.from_workers(workers)
    pool.apply_plan(WorkerActivationPlan.from_world_size(3, 1))
    original_worker = pool.worker_for_rank(2)

    pool.apply_plan(WorkerActivationPlan.from_world_size(3, 3))

    assert pool.worker_for_rank(2) is original_worker
    assert pool.role_for_rank(2) is WorkerRole.ACTIVE
    assert pool.call_worker(2, "execute_model", object()) == "rank-2"
    assert workers[2].calls == ["execute_model"]


def test_worker_wrapper_rejects_execute_model_while_standby():
    wrapper = WorkerWrapperBase(rpc_rank=0, global_rank=0)
    wrapper.worker = FakeWorker(0)  # type: ignore[assignment]
    wrapper.mm_receiver_cache = None

    wrapper.set_worker_role(WorkerRole.STANDBY)

    with pytest.raises(RuntimeError, match="standby"):
        wrapper.execute_model(object())

    assert wrapper.worker.calls == []


def test_worker_wrapper_allows_execute_model_after_wake():
    wrapper = WorkerWrapperBase(rpc_rank=0, global_rank=0)
    wrapper.worker = FakeWorker(0)  # type: ignore[assignment]
    wrapper.mm_receiver_cache = None

    wrapper.set_worker_role(WorkerRole.STANDBY)
    wrapper.set_worker_role(WorkerRole.ACTIVE)

    assert wrapper.execute_model(object()) == "rank-0"
    assert wrapper.worker.calls == ["execute_model"]


def test_worker_wrapper_applies_activation_plan_by_global_rank():
    rank0 = WorkerWrapperBase(rpc_rank=0, global_rank=0)
    rank0.worker = FakeWorker(0)  # type: ignore[assignment]
    rank1 = WorkerWrapperBase(rpc_rank=1, global_rank=1)
    rank1.worker = FakeWorker(1)  # type: ignore[assignment]

    assert rank0.apply_worker_activation_plan((0,)) is WorkerRole.ACTIVE
    assert rank1.apply_worker_activation_plan((0,)) is WorkerRole.STANDBY


def test_worker_wrapper_wake_allows_existing_processes_to_run_collective():
    world_size = 2
    init_method = f"tcp://127.0.0.1:{get_open_port()}"
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    processes = [
        ctx.Process(
            target=_worker_wake_collective_process,
            args=(rank, world_size, init_method, queue),
        )
        for rank in range(world_size)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=60)

    assert all(process.exitcode == 0 for process in processes)
    results = sorted((queue.get(timeout=5) for _ in processes),
                     key=lambda item: item["rank"])
    assert results[0]["standby_role"] == WorkerRole.ACTIVE.value
    assert results[1]["standby_role"] == WorkerRole.STANDBY.value
    assert results[1]["standby_rejected"]
    assert [result["active_role"] for result in results] == [
        WorkerRole.ACTIVE.value,
        WorkerRole.ACTIVE.value,
    ]
    assert [result["collective_result"] for result in results] == [1, 1]
    assert [result["collective_calls"] for result in results] == [1, 1]
