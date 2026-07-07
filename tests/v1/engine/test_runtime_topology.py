# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from collections import deque

import pytest

from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.topology_cache import TopologyDescriptor
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.core_client import AsyncMPClient, SyncMPClient
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.engine.runtime_topology import (
    RuntimeTopologySwitchRequest,
    validate_runtime_topology_switch,
)


def make_config(
    *,
    tp: int = 2,
    pp: int = 1,
    world_size: int = 2,
    backend: str | None = "mp",
    dp: int = 1,
    pcp: int = 1,
    dcp: int = 1,
    lora_config=None,
    speculative_config=None,
    kv_transfer_config=None,
    ec_transfer_config=None,
    kv_offloading_size=None,
    cpu_offload_gb: float = 0.0,
    offload_group_size: int = 0,
    cudagraph_mode=CUDAGraphMode.NONE,
):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            world_size=world_size,
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            data_parallel_size=dp,
            prefill_context_parallel_size=pcp,
            decode_context_parallel_size=dcp,
            distributed_executor_backend=backend,
            enable_expert_parallel=False,
            enable_elastic_ep=False,
        ),
        lora_config=lora_config,
        speculative_config=speculative_config,
        kv_transfer_config=kv_transfer_config,
        ec_transfer_config=ec_transfer_config,
        cache_config=SimpleNamespace(kv_offloading_size=kv_offloading_size),
        offload_config=SimpleNamespace(
            uva=SimpleNamespace(cpu_offload_gb=cpu_offload_gb),
            prefetch=SimpleNamespace(offload_group_size=offload_group_size),
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=cudagraph_mode),
    )


def test_validate_accepts_same_world_size_target_from_env(monkeypatch):
    monkeypatch.setenv("VLLM_PREBUILD_MODEL_PARALLEL_TOPOLOGIES", "tp=1,pp=2")

    plan = validate_runtime_topology_switch(
        make_config(tp=2, pp=1, world_size=2),
        RuntimeTopologySwitchRequest(
            tensor_parallel_size=1,
            pipeline_parallel_size=2,
        ),
    )

    assert plan.previous_topology == TopologyDescriptor(
        world_size=2,
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
    )
    assert plan.target_topology == TopologyDescriptor(
        world_size=2,
        tensor_parallel_size=1,
        pipeline_parallel_size=2,
    )


def test_validate_rejects_target_that_was_not_prebuilt(monkeypatch):
    monkeypatch.setenv("VLLM_PREBUILD_MODEL_PARALLEL_TOPOLOGIES", "")

    with pytest.raises(ValueError, match="prebuilt"):
        validate_runtime_topology_switch(
            make_config(tp=2, pp=1, world_size=2),
            RuntimeTopologySwitchRequest(
                tensor_parallel_size=1,
                pipeline_parallel_size=2,
            ),
        )


@pytest.mark.parametrize(
    ("config_kwargs", "match"),
    [
        ({"backend": "ray"}, "mp"),
        ({"dp": 2, "world_size": 4}, "data parallel"),
        ({"pcp": 2, "world_size": 4}, "context parallel"),
        ({"dcp": 2}, "context parallel"),
        ({"lora_config": object()}, "LoRA"),
        ({"speculative_config": object()}, "speculative"),
        ({"kv_transfer_config": object()}, "KV transfer"),
        ({"ec_transfer_config": object()}, "EC transfer"),
        ({"kv_offloading_size": 1.0}, "KV offload"),
        ({"cpu_offload_gb": 1.0}, "weight offload"),
        ({"offload_group_size": 1}, "weight offload"),
        ({"cudagraph_mode": CUDAGraphMode.PIECEWISE}, "CUDA graph"),
    ],
)
def test_validate_rejects_unsupported_runtime_switch_features(
    monkeypatch, config_kwargs, match
):
    monkeypatch.setenv("VLLM_PREBUILD_MODEL_PARALLEL_TOPOLOGIES", "tp=1,pp=2")

    with pytest.raises(ValueError, match=match):
        validate_runtime_topology_switch(
            make_config(**config_kwargs),
            RuntimeTopologySwitchRequest(
                tensor_parallel_size=1,
                pipeline_parallel_size=2,
            ),
        )


def test_validate_rejects_target_world_size_change(monkeypatch):
    monkeypatch.setenv("VLLM_PREBUILD_MODEL_PARALLEL_TOPOLOGIES", "tp=2,pp=2")

    with pytest.raises(ValueError, match="world size"):
        validate_runtime_topology_switch(
            make_config(tp=2, pp=1, world_size=2),
            RuntimeTopologySwitchRequest(
                tensor_parallel_size=2,
                pipeline_parallel_size=2,
            ),
        )


class FakeSwitchScheduler:

    def __init__(self, events):
        self.events = events
        self.pause_state = PauseState.UNPAUSED
        self.drained_requests = [SimpleNamespace(request_id="req-1")]
        self.added_requests = []
        self.finished_req_ids_dict = None

    def set_pause_state(self, pause_state):
        self.pause_state = pause_state
        self.events.append(("pause_state", pause_state))

    def drain_unfinished_requests_for_recompute(self, *, reset_running_requests):
        self.events.append(("drain", reset_running_requests))
        return self.drained_requests

    def add_request(self, request):
        self.events.append(("add_request", request.request_id))
        self.added_requests.append(request)


class FakeSwitchExecutor:

    def __init__(self, events):
        self.events = events

    def update_runtime_topology_config(self, descriptor):
        self.events.append(("update_worker_config", descriptor))

    def activate_model_parallel_topology(self, descriptor):
        self.events.append(("activate", descriptor))

    def rebuild_model_for_runtime_topology(self):
        self.events.append("rebuild_model")


def test_engine_core_switch_runtime_topology_transaction_order(monkeypatch):
    monkeypatch.setenv("VLLM_PREBUILD_MODEL_PARALLEL_TOPOLOGIES", "tp=1,pp=2")
    events = []
    config = make_config(tp=2, pp=1, world_size=2)
    engine = EngineCore.__new__(EngineCore)
    engine.vllm_config = config
    engine.scheduler = FakeSwitchScheduler(events)
    engine.model_executor = FakeSwitchExecutor(events)

    def initialize_kv_caches(vllm_config):
        events.append(("initialize_kv", vllm_config.parallel_config.tensor_parallel_size,
                       vllm_config.parallel_config.pipeline_parallel_size))
        return "new-kv-config"

    def rebuild_scheduler(kv_cache_config, drained_requests):
        events.append(("rebuild_scheduler", kv_cache_config,
                       [request.request_id for request in drained_requests]))
        engine.scheduler = FakeSwitchScheduler(events)
        for request in drained_requests:
            engine.scheduler.add_request(request)

    engine._initialize_kv_caches = initialize_kv_caches
    engine._rebuild_scheduler_for_runtime_topology = rebuild_scheduler

    result = engine.switch_runtime_topology(
        RuntimeTopologySwitchRequest(
            tensor_parallel_size=1,
            pipeline_parallel_size=2,
        )
    )

    target = TopologyDescriptor(
        world_size=2,
        tensor_parallel_size=1,
        pipeline_parallel_size=2,
    )
    assert result["target"] == {
        "world_size": 2,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 2,
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "data_parallel_size": 1,
    }
    assert events == [
        ("pause_state", PauseState.PAUSED_ALL),
        ("drain", True),
        ("update_worker_config", target),
        ("activate", target),
        "rebuild_model",
        ("initialize_kv", 1, 2),
        ("rebuild_scheduler", "new-kv-config", ["req-1"]),
        ("add_request", "req-1"),
        ("pause_state", PauseState.UNPAUSED),
    ]


def test_engine_core_switch_runtime_topology_clears_batch_queue(monkeypatch):
    monkeypatch.setenv("VLLM_PREBUILD_MODEL_PARALLEL_TOPOLOGIES", "tp=1,pp=2")
    events = []
    config = make_config(tp=2, pp=1, world_size=2)
    engine = EngineCore.__new__(EngineCore)
    engine.vllm_config = config
    engine.scheduler = FakeSwitchScheduler(events)
    engine.model_executor = FakeSwitchExecutor(events)
    engine.batch_queue = deque(["stale-batch"])

    def initialize_kv_caches(vllm_config):
        return "new-kv-config"

    def rebuild_scheduler(kv_cache_config, drained_requests):
        engine.scheduler = FakeSwitchScheduler(events)

    engine._initialize_kv_caches = initialize_kv_caches
    engine._rebuild_scheduler_for_runtime_topology = rebuild_scheduler

    engine.switch_runtime_topology(
        RuntimeTopologySwitchRequest(
            tensor_parallel_size=1,
            pipeline_parallel_size=2,
        )
    )

    assert list(engine.batch_queue) == []


def test_engine_core_switch_runtime_topology_allows_return_to_initial_topology(
    monkeypatch,
):
    monkeypatch.setenv("VLLM_PREBUILD_MODEL_PARALLEL_TOPOLOGIES", "tp=1,pp=2")
    events = []
    config = make_config(tp=2, pp=1, world_size=2)
    engine = EngineCore.__new__(EngineCore)
    engine.vllm_config = config
    engine.scheduler = FakeSwitchScheduler(events)
    engine.model_executor = FakeSwitchExecutor(events)
    engine._initialize_kv_caches = lambda vllm_config: "new-kv-config"

    def rebuild_scheduler(kv_cache_config, drained_requests):
        engine.scheduler = FakeSwitchScheduler(events)

    engine._rebuild_scheduler_for_runtime_topology = rebuild_scheduler

    engine.switch_runtime_topology(
        RuntimeTopologySwitchRequest(
            tensor_parallel_size=1,
            pipeline_parallel_size=2,
        )
    )
    result = engine.switch_runtime_topology(
        RuntimeTopologySwitchRequest(
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
        )
    )

    assert result["target"]["tensor_parallel_size"] == 2
    assert result["target"]["pipeline_parallel_size"] == 1


def test_llm_engine_switch_runtime_topology_delegates_to_core_client():
    calls = []
    engine = LLMEngine.__new__(LLMEngine)
    engine.engine_core = SimpleNamespace(
        switch_runtime_topology=lambda tp, pp: calls.append((tp, pp)) or {
            "target": {
                "tensor_parallel_size": tp,
                "pipeline_parallel_size": pp,
            }
        }
    )

    result = engine.switch_runtime_topology(
        tensor_parallel_size=1,
        pipeline_parallel_size=2,
    )

    assert calls == [(1, 2)]
    assert result["target"]["pipeline_parallel_size"] == 2


def test_sync_mp_client_switch_runtime_topology_sends_msgpack_safe_dict():
    sent = []
    client = SyncMPClient.__new__(SyncMPClient)
    client.call_utility = lambda method, payload: sent.append((method, payload)) or {
        "target": payload
    }

    result = client.switch_runtime_topology(
        tensor_parallel_size=1,
        pipeline_parallel_size=2,
    )

    assert sent == [
        (
            "switch_runtime_topology",
            {"tensor_parallel_size": 1, "pipeline_parallel_size": 2},
        )
    ]
    assert result["target"]["pipeline_parallel_size"] == 2


def test_async_mp_client_switch_runtime_topology_sends_msgpack_safe_dict():
    import asyncio

    sent = []
    client = AsyncMPClient.__new__(AsyncMPClient)

    async def call_utility_async(method, payload):
        sent.append((method, payload))
        return {"target": payload}

    client.call_utility_async = call_utility_async

    result = asyncio.run(
        client.switch_runtime_topology_async(
            tensor_parallel_size=1,
            pipeline_parallel_size=2,
        )
    )

    assert sent == [
        (
            "switch_runtime_topology",
            {"tensor_parallel_size": 1, "pipeline_parallel_size": 2},
        )
    ]
    assert result["target"]["pipeline_parallel_size"] == 2
