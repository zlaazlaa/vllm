# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import traceback
from pathlib import Path

import multiprocess as mp
import pytest
import torch

from vllm.distributed.topology_cache import TopologyDescriptor


def _run_processes(
    *,
    target,
    world_size: int,
    rendezvous_path: str,
    timeout: int,
    start_method: str,
) -> list[str | None]:
    ctx = mp.get_context(start_method)
    queue = ctx.Queue()
    processes = [
        ctx.Process(
            target=target,
            args=(rank, world_size, rendezvous_path, queue),
        )
        for rank in range(world_size)
    ]

    for process in processes:
        process.start()

    errors = [queue.get(timeout=timeout) for _ in processes]

    for process in processes:
        process.join(timeout=timeout)

    for process in processes:
        assert process.exitcode == 0

    return errors


def _runtime_topology_activation_worker(
    rank: int,
    world_size: int,
    rendezvous_path: str,
    queue,
    *,
    backend: str,
) -> None:
    try:
        if backend == "nccl":
            torch.cuda.set_device(rank)
        else:
            os.environ["GLOO_SOCKET_IFNAME"] = "lo"

        from vllm.distributed import parallel_state

        parallel_state.init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method=f"file://{rendezvous_path}",
            local_rank=rank,
            backend=backend,
        )

        new_group_calls = 0
        original_new_group = torch.distributed.new_group

        def counting_new_group(*args, **kwargs):
            nonlocal new_group_calls
            new_group_calls += 1
            return original_new_group(*args, **kwargs)

        torch.distributed.new_group = counting_new_group

        tp_topology = TopologyDescriptor(
            world_size=world_size,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
        )
        pp_topology = TopologyDescriptor(
            world_size=world_size,
            tensor_parallel_size=1,
            pipeline_parallel_size=2,
        )

        parallel_state.prebuild_model_parallel_topologies(
            [tp_topology, pp_topology],
            backend=backend,
        )
        created_count = new_group_calls

        parallel_state.activate_model_parallel_topology(tp_topology)
        assert new_group_calls == created_count
        assert parallel_state.get_tp_group().world_size == 2
        assert parallel_state.get_pp_group().world_size == 1

        parallel_state.activate_model_parallel_topology(pp_topology)
        assert new_group_calls == created_count
        assert parallel_state.get_tp_group().world_size == 1
        assert parallel_state.get_pp_group().world_size == 2

        if backend == "nccl":
            tensor = torch.ones(1, device=f"cuda:{rank}")
            torch.distributed.all_reduce(
                tensor,
                group=parallel_state.get_pp_group().device_group,
            )
        else:
            tensor = torch.ones(1)
            torch.distributed.all_reduce(
                tensor,
                group=parallel_state.get_pp_group().cpu_group,
            )
        assert tensor.item() == world_size

        torch.distributed.new_group = original_new_group
        parallel_state.destroy_model_parallel()
        parallel_state.destroy_distributed_environment()
        queue.put(None)
    except Exception:
        queue.put(traceback.format_exc())


def _gloo_runtime_topology_activation_worker(
    rank: int,
    world_size: int,
    rendezvous_path: str,
    queue,
) -> None:
    _runtime_topology_activation_worker(
        rank,
        world_size,
        rendezvous_path,
        queue,
        backend="gloo",
    )


def _nccl_runtime_topology_activation_worker(
    rank: int,
    world_size: int,
    rendezvous_path: str,
    queue,
) -> None:
    _runtime_topology_activation_worker(
        rank,
        world_size,
        rendezvous_path,
        queue,
        backend="nccl",
    )


def test_gloo_runtime_topology_activation_uses_prebuilt_groups(
    tmp_path: Path,
) -> None:
    errors = _run_processes(
        target=_gloo_runtime_topology_activation_worker,
        world_size=2,
        rendezvous_path=str(tmp_path / "gloo_runtime_topology"),
        timeout=60,
        start_method="fork",
    )

    if all(error is not None and "Operation not permitted" in error
           for error in errors):
        pytest.skip("Gloo process group initialization is not permitted")

    assert errors == [None, None], "\n\n".join(
        error or "<no error>" for error in errors
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Need at least 2 CUDA GPUs to run the NCCL runtime topology test.",
)
def test_nccl_runtime_topology_activation_uses_prebuilt_groups(
    tmp_path: Path,
) -> None:
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

    errors = _run_processes(
        target=_nccl_runtime_topology_activation_worker,
        world_size=2,
        rendezvous_path=str(tmp_path / "nccl_runtime_topology"),
        timeout=120,
        start_method="spawn",
    )

    assert errors == [None, None], "\n\n".join(
        error or "<no error>" for error in errors
    )
