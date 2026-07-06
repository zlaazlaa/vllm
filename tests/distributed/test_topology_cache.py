# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import traceback
from pathlib import Path

import multiprocess as mp
import pytest
import torch
import torch.distributed

from vllm.distributed.topology_cache import (
    TopologyDescriptor,
    TopologyStateCache,
    plan_topology_groups,
)


class FakeGroup:

    def __init__(self, name: str, ranks: list[list[int]]) -> None:
        self.name = name
        self.ranks = ranks
        self.destroyed = False

    def destroy(self) -> None:
        self.destroyed = True


class RecordingGroupBuilder:

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[list[int]]]] = []

    def __call__(self, group_name: str, group_ranks: list[list[int]]) -> FakeGroup:
        ranks = [list(group) for group in group_ranks]
        self.calls.append((group_name, ranks))
        return FakeGroup(group_name, ranks)


def test_plan_world_2_tensor_parallel_layout() -> None:
    descriptor = TopologyDescriptor(
        world_size=2,
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
    )

    layout = plan_topology_groups(descriptor)

    assert layout.tp == [[0, 1]]
    assert layout.pp == [[0], [1]]
    assert layout.dcp == [[0], [1]]
    assert layout.pcp == [[0], [1]]
    assert layout.dp == [[0], [1]]


def test_plan_world_2_pipeline_parallel_layout() -> None:
    descriptor = TopologyDescriptor(
        world_size=2,
        tensor_parallel_size=1,
        pipeline_parallel_size=2,
    )

    layout = plan_topology_groups(descriptor)

    assert layout.tp == [[0], [1]]
    assert layout.pp == [[0, 1]]
    assert layout.dcp == [[0], [1]]
    assert layout.pcp == [[0], [1]]
    assert layout.dp == [[0], [1]]


def test_plan_world_4_tp2_pp2_matches_existing_vllm_layout() -> None:
    descriptor = TopologyDescriptor(
        world_size=4,
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
    )

    layout = plan_topology_groups(descriptor)

    assert layout.tp == [[0, 1], [2, 3]]
    assert layout.pp == [[0, 2], [1, 3]]
    assert layout.dcp == [[0], [1], [2], [3]]
    assert layout.pcp == [[0], [1], [2], [3]]
    assert layout.dp == [[0], [2], [1], [3]]


def test_plan_rejects_world_size_mismatch() -> None:
    with pytest.raises(ValueError, match="world_size"):
        TopologyDescriptor(
            world_size=3,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
        )


def test_plan_rejects_decode_context_larger_than_tensor_parallel() -> None:
    with pytest.raises(ValueError, match="decode_context_parallel_size"):
        TopologyDescriptor(
            world_size=2,
            tensor_parallel_size=1,
            pipeline_parallel_size=2,
            decode_context_parallel_size=2,
        )


def test_plan_rejects_decode_context_that_does_not_divide_tensor_parallel() -> None:
    with pytest.raises(ValueError, match="divide"):
        TopologyDescriptor(
            world_size=6,
            tensor_parallel_size=3,
            pipeline_parallel_size=2,
            decode_context_parallel_size=2,
        )


def test_cache_prebuilds_once_and_activate_does_not_create_groups() -> None:
    builder = RecordingGroupBuilder()
    cache = TopologyStateCache(builder)
    first = TopologyDescriptor(
        world_size=2,
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
    )
    second = TopologyDescriptor(
        world_size=2,
        tensor_parallel_size=1,
        pipeline_parallel_size=2,
    )

    snapshot = cache.prebuild(first)

    assert snapshot.descriptor == first
    assert [call[0] for call in builder.calls] == ["tp", "dcp", "pcp", "pp", "dp"]

    created_count = len(builder.calls)
    assert cache.activate(first) is snapshot
    assert len(builder.calls) == created_count

    assert cache.prebuild(first) is snapshot
    assert len(builder.calls) == created_count

    second_snapshot = cache.prebuild(second)
    assert second_snapshot is not snapshot
    assert len(builder.calls) == created_count * 2


def test_cache_destroy_destroys_all_snapshots() -> None:
    builder = RecordingGroupBuilder()
    cache = TopologyStateCache(builder)
    first = TopologyDescriptor(
        world_size=2,
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
    )
    second = TopologyDescriptor(
        world_size=2,
        tensor_parallel_size=1,
        pipeline_parallel_size=2,
    )

    first_snapshot = cache.prebuild(first)
    second_snapshot = cache.prebuild(second)

    cache.destroy()

    for snapshot in (first_snapshot, second_snapshot):
        assert snapshot.destroyed
        assert snapshot.tp.destroyed
        assert snapshot.dcp.destroyed
        assert snapshot.pcp.destroyed
        assert snapshot.pp.destroyed
        assert snapshot.dp.destroyed


def test_parallel_state_prebuild_activate_and_destroy(monkeypatch) -> None:
    from vllm.distributed import parallel_state

    builder = RecordingGroupBuilder()

    def fake_init_model_parallel_group(
        group_ranks,
        local_rank,
        backend,
        use_message_queue_broadcaster=False,
        group_name=None,
        use_device_communicator=True,
    ):
        return builder(group_name, group_ranks)

    first = TopologyDescriptor(
        world_size=2,
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
    )
    second = TopologyDescriptor(
        world_size=2,
        tensor_parallel_size=1,
        pipeline_parallel_size=2,
    )

    monkeypatch.setattr(parallel_state, "init_model_parallel_group",
                        fake_init_model_parallel_group)
    monkeypatch.setattr(parallel_state, "get_world_group",
                        lambda: type("World", (), {"local_rank": 0})())
    monkeypatch.setattr(parallel_state.torch.distributed, "get_backend",
                        lambda group: "gloo")
    monkeypatch.setattr(parallel_state, "_TP", None)
    monkeypatch.setattr(parallel_state, "_DCP", None)
    monkeypatch.setattr(parallel_state, "_PCP", None)
    monkeypatch.setattr(parallel_state, "_PP", None)
    monkeypatch.setattr(parallel_state, "_DP", None)
    monkeypatch.setattr(parallel_state, "_MODEL_PARALLEL_TOPOLOGY_CACHE", None)

    parallel_state.prebuild_model_parallel_topologies([first, second], backend="gloo")

    assert [call[0] for call in builder.calls] == [
        "tp", "dcp", "pcp", "pp", "dp",
        "tp", "dcp", "pcp", "pp", "dp",
    ]

    created_count = len(builder.calls)
    parallel_state.activate_model_parallel_topology(first)

    assert parallel_state.get_tp_group().ranks == [[0, 1]]
    assert parallel_state.get_pp_group().ranks == [[0], [1]]
    assert len(builder.calls) == created_count

    parallel_state.activate_model_parallel_topology(second)

    assert parallel_state.get_tp_group().ranks == [[0], [1]]
    assert parallel_state.get_pp_group().ranks == [[0, 1]]
    assert len(builder.calls) == created_count

    first_groups = (
        parallel_state.get_dcp_group(),
        parallel_state.get_pcp_group(),
        parallel_state.get_pp_group(),
        parallel_state.get_dp_group(),
    )

    parallel_state.destroy_model_parallel_topology_cache()

    assert all(group.destroyed for group in first_groups)
    assert parallel_state._MODEL_PARALLEL_TOPOLOGY_CACHE is None
    assert parallel_state._TP is None
    assert parallel_state._DCP is None
    assert parallel_state._PCP is None
    assert parallel_state._PP is None
    assert parallel_state._DP is None


def test_destroy_model_parallel_destroys_topology_cache(monkeypatch) -> None:
    from vllm.distributed import parallel_state

    builder = RecordingGroupBuilder()

    def fake_init_model_parallel_group(
        group_ranks,
        local_rank,
        backend,
        use_message_queue_broadcaster=False,
        group_name=None,
        use_device_communicator=True,
    ):
        return builder(group_name, group_ranks)

    first = TopologyDescriptor(
        world_size=2,
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
    )
    second = TopologyDescriptor(
        world_size=2,
        tensor_parallel_size=1,
        pipeline_parallel_size=2,
    )

    monkeypatch.setattr(
        parallel_state, "init_model_parallel_group", fake_init_model_parallel_group
    )
    monkeypatch.setattr(
        parallel_state,
        "get_world_group",
        lambda: type("World", (), {"local_rank": 0})(),
    )
    monkeypatch.setattr(parallel_state, "_TP", None)
    monkeypatch.setattr(parallel_state, "_DCP", None)
    monkeypatch.setattr(parallel_state, "_PCP", None)
    monkeypatch.setattr(parallel_state, "_PP", None)
    monkeypatch.setattr(parallel_state, "_DP", None)
    monkeypatch.setattr(parallel_state, "_EP", None)
    monkeypatch.setattr(parallel_state, "_EPLB", None)
    monkeypatch.setattr(parallel_state, "_MODEL_PARALLEL_TOPOLOGY_CACHE", None)

    parallel_state.prebuild_model_parallel_topologies([first, second], backend="gloo")
    parallel_state.activate_model_parallel_topology(first)
    cache = parallel_state._MODEL_PARALLEL_TOPOLOGY_CACHE
    assert cache is not None
    snapshots = list(cache._snapshots.values())

    parallel_state.destroy_model_parallel()

    assert parallel_state._MODEL_PARALLEL_TOPOLOGY_CACHE is None
    assert all(snapshot.destroyed for snapshot in snapshots)


def _gloo_cache_worker(
    rank: int, world_size: int, rendezvous_path: str, queue
) -> None:
    try:
        from vllm.distributed.parallel_state import GroupCoordinator

        os.environ["GLOO_SOCKET_IFNAME"] = "lo"
        init_method = f"file://{rendezvous_path}"
        torch.distributed.init_process_group(
            backend="gloo",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )

        new_group_calls = 0
        original_new_group = torch.distributed.new_group

        def counting_new_group(*args, **kwargs):
            nonlocal new_group_calls
            new_group_calls += 1
            return original_new_group(*args, **kwargs)

        torch.distributed.new_group = counting_new_group

        def build_group(group_name: str, group_ranks: list[list[int]]):
            return GroupCoordinator(
                group_ranks=group_ranks,
                local_rank=rank,
                torch_distributed_backend="gloo",
                use_device_communicator=False,
                group_name=f"test_{group_name}",
            )

        first = TopologyDescriptor(
            world_size=world_size,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
        )
        second = TopologyDescriptor(
            world_size=world_size,
            tensor_parallel_size=1,
            pipeline_parallel_size=2,
        )

        cache = TopologyStateCache(build_group)
        cache.prebuild(first)
        cache.prebuild(second)

        created_count = new_group_calls

        first_snapshot = cache.activate(first)
        assert new_group_calls == created_count
        tensor = torch.ones(1)
        torch.distributed.all_reduce(tensor, group=first_snapshot.tp.cpu_group)
        assert tensor.item() == world_size

        second_snapshot = cache.activate(second)
        assert new_group_calls == created_count
        tensor = torch.ones(1)
        torch.distributed.all_reduce(tensor, group=second_snapshot.pp.cpu_group)
        assert tensor.item() == world_size

        cache.destroy()
        torch.distributed.new_group = original_new_group
        torch.distributed.destroy_process_group()
        queue.put(None)
    except Exception:
        queue.put(traceback.format_exc())


def test_gloo_cache_activation_does_not_create_new_groups(tmp_path: Path) -> None:
    world_size = 2
    rendezvous_path = str(tmp_path / "gloo_rendezvous")
    queue = mp.Queue()
    processes = [
        mp.Process(
            target=_gloo_cache_worker,
            args=(rank, world_size, rendezvous_path, queue),
        )
        for rank in range(world_size)
    ]

    for process in processes:
        process.start()

    errors = [queue.get(timeout=60) for _ in processes]

    for process in processes:
        process.join(timeout=60)

    for process in processes:
        assert process.exitcode == 0

    if all(error is not None and "Operation not permitted" in error
           for error in errors):
        pytest.skip("Gloo process group initialization is not permitted")

    assert errors == [None, None]


def _nccl_cache_worker(
    rank: int, world_size: int, rendezvous_path: str, queue
) -> None:
    try:
        from vllm.distributed.parallel_state import GroupCoordinator

        torch.cuda.set_device(rank)
        init_method = f"file://{rendezvous_path}"
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )

        new_group_calls = 0
        original_new_group = torch.distributed.new_group

        def counting_new_group(*args, **kwargs):
            nonlocal new_group_calls
            new_group_calls += 1
            return original_new_group(*args, **kwargs)

        torch.distributed.new_group = counting_new_group

        def build_group(group_name: str, group_ranks: list[list[int]]):
            return GroupCoordinator(
                group_ranks=group_ranks,
                local_rank=rank,
                torch_distributed_backend="nccl",
                use_device_communicator=False,
                group_name=f"test_nccl_{group_name}",
            )

        first = TopologyDescriptor(
            world_size=world_size,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
        )
        second = TopologyDescriptor(
            world_size=world_size,
            tensor_parallel_size=1,
            pipeline_parallel_size=2,
        )

        cache = TopologyStateCache(build_group)
        cache.prebuild(first)
        cache.prebuild(second)

        created_count = new_group_calls

        first_snapshot = cache.activate(first)
        assert new_group_calls == created_count
        tensor = torch.ones(1, device=f"cuda:{rank}")
        torch.distributed.all_reduce(tensor, group=first_snapshot.tp.device_group)
        assert tensor.item() == world_size

        second_snapshot = cache.activate(second)
        assert new_group_calls == created_count
        tensor = torch.ones(1, device=f"cuda:{rank}")
        torch.distributed.all_reduce(tensor, group=second_snapshot.pp.device_group)
        assert tensor.item() == world_size

        cache.destroy()
        torch.distributed.new_group = original_new_group
        torch.distributed.destroy_process_group()
        queue.put(None)
    except Exception:
        queue.put(traceback.format_exc())


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Need at least 2 CUDA GPUs to run the NCCL topology cache test.",
)
def test_nccl_cache_activation_does_not_create_new_groups(tmp_path: Path) -> None:
    world_size = 2
    rendezvous_path = str(tmp_path / "nccl_rendezvous")
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    processes = [
        ctx.Process(
            target=_nccl_cache_worker,
            args=(rank, world_size, rendezvous_path, queue),
        )
        for rank in range(world_size)
    ]

    for process in processes:
        process.start()

    errors = [queue.get(timeout=120) for _ in processes]

    for process in processes:
        process.join(timeout=120)

    for process in processes:
        assert process.exitcode == 0

    assert errors == [None, None]
