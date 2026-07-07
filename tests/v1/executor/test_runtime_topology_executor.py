# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.distributed.topology_cache import TopologyDescriptor
from vllm.v1.executor.abstract import Executor
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.worker.gpu_worker import Worker


class RecordingExecutor(Executor):

    def _init_executor(self) -> None:
        self.calls = []

    def collective_rpc(
        self,
        method,
        timeout=None,
        args=(),
        kwargs=None,
        non_block: bool = False,
    ):
        self.calls.append((method, timeout, args, kwargs, non_block))
        return []

    def check_health(self) -> None:
        return None


class FakeModelRunner:

    def __init__(self) -> None:
        self.calls = []

    def clear_runtime_state_for_topology_rebuild(
        self,
        *,
        clear_model: bool,
    ) -> None:
        self.calls.append(("clear_runtime", clear_model))


def _fake_executor() -> RecordingExecutor:
    config = SimpleNamespace(
        model_config=None,
        cache_config=None,
        lora_config=None,
        load_config=None,
        parallel_config=None,
        scheduler_config=None,
        device_config=None,
        speculative_config=None,
        observability_config=None,
    )
    return RecordingExecutor(config)


def test_executor_delegates_runtime_topology_primitives_to_workers():
    executor = _fake_executor()
    descriptor = TopologyDescriptor(
        world_size=2,
        tensor_parallel_size=1,
        pipeline_parallel_size=2,
    )

    executor.update_runtime_topology_config(descriptor)
    executor.activate_model_parallel_topology(descriptor)
    executor.rebuild_model_for_runtime_topology()
    executor.clear_runtime_kv_state()

    assert executor.calls == [
        (
            "update_runtime_topology_config",
            None,
            (descriptor,),
            None,
            False,
        ),
        (
            "activate_model_parallel_topology",
            None,
            (descriptor,),
            None,
            False,
        ),
        ("rebuild_model_for_runtime_topology", None, (), None, False),
        ("clear_runtime_kv_state", None, (), None, False),
    ]


def test_multiproc_executor_updates_runtime_output_rank():
    executor = MultiprocExecutor.__new__(MultiprocExecutor)
    executor.parallel_config = SimpleNamespace(
        world_size=2,
        local_world_size=2,
        nnodes_within_dp=1,
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        data_parallel_size=1,
    )
    executor.vllm_config = SimpleNamespace(parallel_config=executor.parallel_config)
    executor.world_size = 2
    executor.output_rank = executor._get_output_rank()
    calls = []

    def collective_rpc(method, timeout=None, args=(), kwargs=None, non_block=False):
        calls.append((method, args))
        return []

    executor.collective_rpc = collective_rpc
    descriptor = TopologyDescriptor(
        world_size=2,
        tensor_parallel_size=1,
        pipeline_parallel_size=2,
    )

    executor.update_runtime_topology_config(descriptor)

    assert calls == [("update_runtime_topology_config", (descriptor,))]
    assert executor.parallel_config.tensor_parallel_size == 1
    assert executor.parallel_config.pipeline_parallel_size == 2
    assert executor.output_rank == 1


def test_gpu_worker_activates_prebuilt_topology(monkeypatch):
    worker = Worker.__new__(Worker)
    descriptor = TopologyDescriptor(
        world_size=2,
        tensor_parallel_size=1,
        pipeline_parallel_size=2,
    )
    activated = []

    monkeypatch.setattr(
        "vllm.v1.worker.gpu_worker.activate_model_parallel_topology",
        lambda item: activated.append(item),
    )

    worker.activate_model_parallel_topology(descriptor)

    assert activated == [descriptor]


def test_gpu_worker_rebuild_model_clears_runtime_state_before_loading():
    worker = Worker.__new__(Worker)
    runner = FakeModelRunner()
    worker.model_runner = runner
    calls = []

    def init_model_runner():
        calls.append("init_model_runner")

    def load_model():
        calls.append("load_model")

    worker._init_model_runner = init_model_runner
    worker.load_model = load_model

    worker.rebuild_model_for_runtime_topology()

    assert runner.calls == [("clear_runtime", True)]
    assert calls == ["init_model_runner", "load_model"]


def test_gpu_worker_clear_runtime_kv_state_keeps_model_weights():
    worker = Worker.__new__(Worker)
    runner = FakeModelRunner()
    worker.model_runner = runner

    worker.clear_runtime_kv_state()

    assert runner.calls == [("clear_runtime", False)]


def test_gpu_worker_updates_runtime_topology_config():
    worker = Worker.__new__(Worker)
    descriptor = TopologyDescriptor(
        world_size=2,
        tensor_parallel_size=1,
        pipeline_parallel_size=2,
    )
    parallel_config = SimpleNamespace(
        world_size=2,
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        data_parallel_size=1,
    )
    worker.parallel_config = parallel_config
    worker.vllm_config = SimpleNamespace(parallel_config=parallel_config)

    worker.update_runtime_topology_config(descriptor)

    assert parallel_config.tensor_parallel_size == 1
    assert parallel_config.pipeline_parallel_size == 2
