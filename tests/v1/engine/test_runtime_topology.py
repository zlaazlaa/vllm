# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import Future
from queue import Queue
from types import SimpleNamespace
from collections import deque

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.topology_cache import TopologyDescriptor
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.engine.core import EngineCore, EngineCoreProc, EngineShutdownState
from vllm.v1.engine.core_client import AsyncMPClient, SyncMPClient
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.engine.runtime_topology import (
    RuntimeTopologySwitchRequest,
    RuntimeTopologyWorkload,
    recommend_runtime_topology,
    validate_runtime_topology_switch,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
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


def make_kv_config(*, num_blocks: int = 8) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=[f"model.layers.{i}.self_attn" for i in range(4)],
                kv_cache_spec=FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=4,
                    head_size=64,
                    dtype=torch.float16,
                ),
            )
        ],
    )


def make_runtime_topology_engine_core_proc() -> EngineCoreProc:
    proc = EngineCoreProc.__new__(EngineCoreProc)
    proc.input_queue = Queue()
    proc.aborts_queue = Queue()
    proc.output_queue = Queue()
    proc.batch_queue = []
    proc.engines_running = False
    proc.scheduler = SimpleNamespace(has_requests=lambda: False)
    proc.shutdown_state = EngineShutdownState.RUNNING
    proc.process_input_queue_block = False
    proc._idle_state_callbacks = []
    return proc


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


@pytest.mark.parametrize(
    "cudagraph_mode",
    [CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE],
)
def test_validate_accepts_cuda_graph_recapture_mode(monkeypatch, cudagraph_mode):
    monkeypatch.setenv("VLLM_PREBUILD_MODEL_PARALLEL_TOPOLOGIES", "tp=1,pp=2")

    plan = validate_runtime_topology_switch(
        make_config(
            tp=2,
            pp=1,
            world_size=2,
            cudagraph_mode=cudagraph_mode,
        ),
        RuntimeTopologySwitchRequest(
            tensor_parallel_size=1,
            pipeline_parallel_size=2,
        ),
    )

    assert plan.target_topology.tensor_parallel_size == 1
    assert plan.target_topology.pipeline_parallel_size == 2


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


def test_validate_rejects_invalid_runtime_kv_migration_batch_size(monkeypatch):
    monkeypatch.setenv("VLLM_PREBUILD_MODEL_PARALLEL_TOPOLOGIES", "tp=1,pp=2")

    with pytest.raises(ValueError, match="max_kv_migration_blocks_per_step"):
        validate_runtime_topology_switch(
            make_config(tp=2, pp=1, world_size=2),
            RuntimeTopologySwitchRequest(
                tensor_parallel_size=1,
                pipeline_parallel_size=2,
                max_kv_migration_blocks_per_step=0,
            ),
        )


def test_validate_accepts_runtime_kv_migration_data_plane(monkeypatch):
    monkeypatch.setenv("VLLM_PREBUILD_MODEL_PARALLEL_TOPOLOGIES", "tp=1,pp=2")

    plan = validate_runtime_topology_switch(
        make_config(tp=2, pp=1, world_size=2),
        RuntimeTopologySwitchRequest(
            tensor_parallel_size=1,
            pipeline_parallel_size=2,
            kv_migration_data_plane="p2p",
        ),
    )

    assert plan.kv_migration_data_plane == "p2p"


def test_validate_rejects_invalid_runtime_kv_migration_data_plane(monkeypatch):
    monkeypatch.setenv("VLLM_PREBUILD_MODEL_PARALLEL_TOPOLOGIES", "tp=1,pp=2")

    with pytest.raises(ValueError, match="kv_migration_data_plane"):
        validate_runtime_topology_switch(
            make_config(tp=2, pp=1, world_size=2),
            RuntimeTopologySwitchRequest(
                tensor_parallel_size=1,
                pipeline_parallel_size=2,
                kv_migration_data_plane="unknown",
            ),
        )


def test_recommend_runtime_topology_prefers_tp_for_low_concurrency(monkeypatch):
    monkeypatch.setenv(
        "VLLM_PREBUILD_MODEL_PARALLEL_TOPOLOGIES",
        "tp=4,pp=1;tp=2,pp=2",
    )

    recommendation = recommend_runtime_topology(
        make_config(tp=1, pp=4, world_size=4),
        RuntimeTopologyWorkload(num_running_requests=0, num_waiting_requests=1),
    )

    assert recommendation.target_topology == TopologyDescriptor(
        world_size=4,
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
    )
    assert recommendation.reason == "low_concurrency"


def test_recommend_runtime_topology_prefers_pp_for_high_concurrency(monkeypatch):
    monkeypatch.setenv(
        "VLLM_PREBUILD_MODEL_PARALLEL_TOPOLOGIES",
        "tp=1,pp=4;tp=2,pp=2",
    )

    recommendation = recommend_runtime_topology(
        make_config(tp=4, pp=1, world_size=4),
        RuntimeTopologyWorkload(num_running_requests=3, num_waiting_requests=3),
    )

    assert recommendation.target_topology == TopologyDescriptor(
        world_size=4,
        tensor_parallel_size=1,
        pipeline_parallel_size=4,
    )
    assert recommendation.reason == "high_concurrency"


def test_recommend_runtime_topology_only_uses_prebuilt_candidates(monkeypatch):
    monkeypatch.setenv(
        "VLLM_PREBUILD_MODEL_PARALLEL_TOPOLOGIES",
        "tp=2,pp=2",
    )

    recommendation = recommend_runtime_topology(
        make_config(tp=4, pp=1, world_size=4),
        RuntimeTopologyWorkload(num_running_requests=5, num_waiting_requests=5),
    )

    assert recommendation.target_topology == TopologyDescriptor(
        world_size=4,
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
    )
    assert recommendation.reason == "high_concurrency"


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

    def drain_unfinished_requests_for_runtime_kv_migration(self):
        self.events.append("drain_migrate")
        return self.drained_requests

    def restore_runtime_kv_blocks_for_migration(
        self,
        request_block_ids,
        block_mapping,
    ):
        self.events.append(
            (
                "restore_kv_blocks",
                dict(request_block_ids),
                dict(block_mapping),
            )
        )

    def collect_runtime_kv_block_ids(self):
        return {}

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

    def snapshot_runtime_kv_caches_for_topology_migration(self):
        self.events.append("snapshot_kv")

    def clear_runtime_kv_migration_snapshot(self):
        self.events.append("clear_kv_snapshot")

    def migrate_runtime_kv_cache_for_topology(
        self,
        *,
        plan,
        block_mapping,
        max_blocks_per_step=1,
    ):
        self.events.append(
            (
                "migrate_kv",
                plan.target_topology,
                dict(block_mapping),
                max_blocks_per_step,
            )
        )
        return {
            "migration_steps": 2,
            "tensor_copies": 6,
            "source_shards": 2,
        }

    def migrate_runtime_kv_cache_for_topology_p2p(
        self,
        *,
        plan,
        block_mapping,
        max_blocks_per_step=1,
    ):
        self.events.append(
            (
                "migrate_kv_p2p",
                plan.target_topology,
                dict(block_mapping),
                max_blocks_per_step,
            )
        )
        return {
            "migration_steps": 2,
            "tensor_copies": 6,
            "p2p_sends": 1,
            "p2p_recvs": 1,
        }


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
        ("update_worker_config", target),
        ("activate", target),
        "rebuild_model",
        ("initialize_kv", 1, 2),
        ("drain", True),
        ("rebuild_scheduler", "new-kv-config", ["req-1"]),
        ("add_request", "req-1"),
        ("pause_state", PauseState.UNPAUSED),
    ]


def test_engine_core_switch_runtime_topology_waits_for_pending_pause(
    monkeypatch,
):
    monkeypatch.setenv("VLLM_PREBUILD_MODEL_PARALLEL_TOPOLOGIES", "tp=1,pp=2")
    events = []
    config = make_config(tp=2, pp=1, world_size=2)
    engine = EngineCore.__new__(EngineCore)
    engine.vllm_config = config
    engine.scheduler = FakeSwitchScheduler(events)
    engine.model_executor = FakeSwitchExecutor(events)
    pause_future: Future[None] = Future()

    def pause_scheduler(*, mode, clear_cache):
        events.append(("pause", mode, clear_cache))
        return pause_future

    def collect_runtime_kv_block_ids():
        events.append("collect_kv_blocks")
        return {}

    def initialize_kv_caches(vllm_config):
        events.append("initialize_kv")
        return "new-kv-config"

    def rebuild_scheduler(kv_cache_config, drained_requests):
        events.append("rebuild_scheduler")
        engine.scheduler = FakeSwitchScheduler(events)

    engine.pause_scheduler = pause_scheduler
    engine.scheduler.collect_runtime_kv_block_ids = collect_runtime_kv_block_ids
    engine._initialize_kv_caches = initialize_kv_caches
    engine._rebuild_scheduler_for_runtime_topology = rebuild_scheduler

    result_future = engine.switch_runtime_topology(
        RuntimeTopologySwitchRequest(
            tensor_parallel_size=1,
            pipeline_parallel_size=2,
        )
    )

    assert isinstance(result_future, Future)
    assert not result_future.done()
    assert events == [("pause", "keep", False)]

    pause_future.set_result(None)
    result = result_future.result(timeout=0)

    assert result["target"]["tensor_parallel_size"] == 1
    assert result["target"]["pipeline_parallel_size"] == 2
    assert "collect_kv_blocks" in events
    assert events.index("collect_kv_blocks") > events.index(
        ("pause", "keep", False)
    )


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


def test_count_live_runtime_kv_blocks_excludes_null_block():
    assert EngineCore._count_live_runtime_kv_blocks({
        "req-0": [0, 0],
        "req-1": [0, 3, 5],
        "req-2": [5],
    }) == 2


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


def test_engine_core_switch_runtime_topology_prepares_kv_migration_plan(
    monkeypatch,
):
    monkeypatch.setenv("VLLM_PREBUILD_MODEL_PARALLEL_TOPOLOGIES", "tp=1,pp=2")
    events = []
    config = make_config(tp=2, pp=1, world_size=2)
    engine = EngineCore.__new__(EngineCore)
    engine.vllm_config = config
    engine.scheduler = FakeSwitchScheduler(events)
    engine.scheduler.collect_runtime_kv_block_ids = lambda: {
        "req-0": [2, 4],
        "req-1": [4, 7],
    }
    engine.model_executor = FakeSwitchExecutor(events)
    engine._initialize_kv_caches = lambda vllm_config: make_kv_config(
        num_blocks=8
    )

    def rebuild_scheduler(kv_cache_config, drained_requests):
        engine.scheduler = FakeSwitchScheduler(events)

    engine._rebuild_scheduler_for_runtime_topology = rebuild_scheduler

    result = engine.switch_runtime_topology(
        RuntimeTopologySwitchRequest(
            tensor_parallel_size=1,
            pipeline_parallel_size=2,
        )
    )

    assert {
        "migration_steps",
        "tensor_copies",
        "source_shards",
    } <= result["kv_cache_migration"].keys()
    assert {
        key: result["kv_cache_migration"][key]
        for key in ("policy", "reason", "live_blocks", "target_num_blocks")
    } == {
        "policy": "migrate",
        "reason": "capacity_available",
        "live_blocks": 3,
        "target_num_blocks": 8,
    }
    assert engine._runtime_kv_migration_preparation["block_mapping"] == {
        2: 2,
        4: 4,
        7: 7,
    }
    assert "snapshot_kv" in events
    assert "drain_migrate" in events


def test_engine_core_switch_runtime_topology_executes_kv_migration(
    monkeypatch,
):
    monkeypatch.setenv("VLLM_PREBUILD_MODEL_PARALLEL_TOPOLOGIES", "tp=1,pp=2")
    events = []
    config = make_config(tp=2, pp=1, world_size=2)
    engine = EngineCore.__new__(EngineCore)
    engine.vllm_config = config
    engine.scheduler = FakeSwitchScheduler(events)
    engine.scheduler.collect_runtime_kv_block_ids = lambda: {
        "req-0": [2, 4],
        "req-1": [4, 7],
    }
    engine.model_executor = FakeSwitchExecutor(events)
    engine._initialize_kv_caches = lambda vllm_config: make_kv_config(
        num_blocks=8
    )

    def rebuild_scheduler(kv_cache_config, drained_requests):
        engine.scheduler = FakeSwitchScheduler(events)

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
    assert (
        "migrate_kv",
        target,
        {2: 2, 4: 4, 7: 7},
        1,
    ) in events
    assert result["kv_cache_migration"]["migration_steps"] == 2
    assert result["kv_cache_migration"]["tensor_copies"] == 6
    assert result["kv_cache_migration"]["source_shards"] == 2
    assert result["kv_cache_migration"]["request_state"] == "migrated"
    assert (
        "restore_kv_blocks",
        {"req-0": [2, 4], "req-1": [4, 7]},
        {2: 2, 4: 4, 7: 7},
    ) in events


def test_engine_core_switch_runtime_topology_uses_requested_kv_migration_batch_size(
    monkeypatch,
):
    monkeypatch.setenv("VLLM_PREBUILD_MODEL_PARALLEL_TOPOLOGIES", "tp=1,pp=2")
    events = []
    config = make_config(tp=2, pp=1, world_size=2)
    engine = EngineCore.__new__(EngineCore)
    engine.vllm_config = config
    engine.scheduler = FakeSwitchScheduler(events)
    engine.scheduler.collect_runtime_kv_block_ids = lambda: {
        "req-0": [2, 4, 7],
    }
    engine.model_executor = FakeSwitchExecutor(events)
    engine._initialize_kv_caches = lambda vllm_config: make_kv_config(
        num_blocks=8
    )

    def rebuild_scheduler(kv_cache_config, drained_requests):
        engine.scheduler = FakeSwitchScheduler(events)

    engine._rebuild_scheduler_for_runtime_topology = rebuild_scheduler

    result = engine.switch_runtime_topology(
        RuntimeTopologySwitchRequest(
            tensor_parallel_size=1,
            pipeline_parallel_size=2,
            max_kv_migration_blocks_per_step=3,
        )
    )

    target = TopologyDescriptor(
        world_size=2,
        tensor_parallel_size=1,
        pipeline_parallel_size=2,
    )
    assert (
        "migrate_kv",
        target,
        {2: 2, 4: 4, 7: 7},
        3,
    ) in events
    assert result["kv_cache_migration"]["max_blocks_per_step"] == 3


def test_engine_core_switch_runtime_topology_uses_requested_kv_data_plane(
    monkeypatch,
):
    monkeypatch.setenv("VLLM_PREBUILD_MODEL_PARALLEL_TOPOLOGIES", "tp=1,pp=2")
    events = []
    config = make_config(tp=2, pp=1, world_size=2)
    engine = EngineCore.__new__(EngineCore)
    engine.vllm_config = config
    engine.scheduler = FakeSwitchScheduler(events)
    engine.scheduler.collect_runtime_kv_block_ids = lambda: {
        "req-0": [2, 4, 7],
    }
    engine.model_executor = FakeSwitchExecutor(events)
    engine._initialize_kv_caches = lambda vllm_config: make_kv_config(
        num_blocks=8
    )

    def rebuild_scheduler(kv_cache_config, drained_requests):
        engine.scheduler = FakeSwitchScheduler(events)

    engine._rebuild_scheduler_for_runtime_topology = rebuild_scheduler

    result = engine.switch_runtime_topology(
        RuntimeTopologySwitchRequest(
            tensor_parallel_size=1,
            pipeline_parallel_size=2,
            kv_migration_data_plane="p2p",
        )
    )

    target = TopologyDescriptor(
        world_size=2,
        tensor_parallel_size=1,
        pipeline_parallel_size=2,
    )
    assert (
        "migrate_kv_p2p",
        target,
        {2: 2, 4: 4, 7: 7},
        1,
    ) in events
    assert not any(
        event[0] == "migrate_kv"
        for event in events
        if isinstance(event, tuple)
    )
    assert result["kv_cache_migration"]["data_plane"] == "p2p"
    assert result["kv_cache_migration"]["p2p_sends"] == 1
    assert result["kv_cache_migration"]["p2p_recvs"] == 1


@pytest.mark.parametrize(
    ("load_format", "expected_uses_host_weight_store"),
    [
        ("host_weight_store", True),
        ("auto", False),
    ],
)
def test_engine_core_switch_runtime_topology_reports_model_materialization_source(
    monkeypatch,
    load_format,
    expected_uses_host_weight_store,
):
    monkeypatch.setenv("VLLM_PREBUILD_MODEL_PARALLEL_TOPOLOGIES", "tp=1,pp=2")
    events = []
    config = make_config(tp=2, pp=1, world_size=2)
    config.load_config = SimpleNamespace(
        load_format=load_format,
        model_loader_extra_config={"metadata_path": "/tmp/shared-weights.json"},
    )
    engine = EngineCore.__new__(EngineCore)
    engine.vllm_config = config
    engine.scheduler = FakeSwitchScheduler(events)
    engine.model_executor = FakeSwitchExecutor(events)
    engine._initialize_kv_caches = lambda vllm_config: make_kv_config()

    def rebuild_scheduler(kv_cache_config, drained_requests):
        engine.scheduler = FakeSwitchScheduler(events)

    engine._rebuild_scheduler_for_runtime_topology = rebuild_scheduler

    result = engine.switch_runtime_topology(
        RuntimeTopologySwitchRequest(
            tensor_parallel_size=1,
            pipeline_parallel_size=2,
        )
    )

    assert result["model_materialization"] == {
        "load_format": load_format,
        "uses_host_weight_store": expected_uses_host_weight_store,
    }


def test_engine_core_switch_runtime_topology_reports_step_timing(monkeypatch):
    monkeypatch.setenv("VLLM_PREBUILD_MODEL_PARALLEL_TOPOLOGIES", "tp=1,pp=2")
    events = []
    config = make_config(tp=2, pp=1, world_size=2)
    engine = EngineCore.__new__(EngineCore)
    engine.vllm_config = config
    engine.scheduler = FakeSwitchScheduler(events)
    engine.model_executor = FakeSwitchExecutor(events)
    engine._initialize_kv_caches = lambda vllm_config: make_kv_config()

    def rebuild_scheduler(kv_cache_config, drained_requests):
        engine.scheduler = FakeSwitchScheduler(events)

    engine._rebuild_scheduler_for_runtime_topology = rebuild_scheduler

    result = engine.switch_runtime_topology(
        RuntimeTopologySwitchRequest(
            tensor_parallel_size=1,
            pipeline_parallel_size=2,
        )
    )

    timing = result["runtime_switch_timing"]
    assert set(timing) == {
        "total_seconds",
        "communication_activate_seconds",
        "model_rebuild_seconds",
        "kv_cache_initialize_seconds",
        "kv_cache_migration_seconds",
        "scheduler_rebuild_seconds",
    }
    assert all(value >= 0.0 for value in timing.values())
    measured_steps = [
        "communication_activate_seconds",
        "model_rebuild_seconds",
        "kv_cache_initialize_seconds",
        "kv_cache_migration_seconds",
        "scheduler_rebuild_seconds",
    ]
    assert timing["total_seconds"] + 1e-6 >= sum(
        timing[key] for key in measured_steps
    )


def test_engine_core_switch_runtime_topology_clears_kv_snapshot_on_recompute(
    monkeypatch,
):
    monkeypatch.setenv("VLLM_PREBUILD_MODEL_PARALLEL_TOPOLOGIES", "tp=1,pp=2")
    events = []
    config = make_config(tp=2, pp=1, world_size=2)
    engine = EngineCore.__new__(EngineCore)
    engine.vllm_config = config
    engine.scheduler = FakeSwitchScheduler(events)
    engine.scheduler.collect_runtime_kv_block_ids = lambda: {
        "req-0": [2, 4],
        "req-1": [6],
    }
    engine.model_executor = FakeSwitchExecutor(events)
    engine._initialize_kv_caches = lambda vllm_config: make_kv_config(
        num_blocks=2
    )

    def rebuild_scheduler(kv_cache_config, drained_requests):
        engine.scheduler = FakeSwitchScheduler(events)

    engine._rebuild_scheduler_for_runtime_topology = rebuild_scheduler

    result = engine.switch_runtime_topology(
        RuntimeTopologySwitchRequest(
            tensor_parallel_size=1,
            pipeline_parallel_size=2,
        )
    )

    assert result["kv_cache_migration"]["policy"] == "recompute"
    assert result["kv_cache_migration"]["reason"] == (
        "insufficient_target_kv_capacity"
    )
    assert "snapshot_kv" in events
    assert "clear_kv_snapshot" in events


def test_engine_core_switch_runtime_topology_falls_back_when_kv_migration_fails(
    monkeypatch,
):
    monkeypatch.setenv("VLLM_PREBUILD_MODEL_PARALLEL_TOPOLOGIES", "tp=1,pp=2")
    events = []
    config = make_config(tp=2, pp=1, world_size=2)
    engine = EngineCore.__new__(EngineCore)
    engine.vllm_config = config
    engine.scheduler = FakeSwitchScheduler(events)
    engine.scheduler.collect_runtime_kv_block_ids = lambda: {
        "req-0": [2, 4],
        "req-1": [4, 7],
    }
    engine.model_executor = FakeSwitchExecutor(events)

    def fail_migration(**kwargs):
        events.append("migrate_kv_failed")
        raise RuntimeError("copy failed")

    engine.model_executor.migrate_runtime_kv_cache_for_topology = fail_migration
    engine._initialize_kv_caches = lambda vllm_config: make_kv_config(
        num_blocks=8
    )

    def rebuild_scheduler(kv_cache_config, drained_requests):
        engine.scheduler = FakeSwitchScheduler(events)

    engine._rebuild_scheduler_for_runtime_topology = rebuild_scheduler

    result = engine.switch_runtime_topology(
        RuntimeTopologySwitchRequest(
            tensor_parallel_size=1,
            pipeline_parallel_size=2,
        )
    )

    assert result["kv_cache_migration"]["policy"] == "recompute"
    assert result["kv_cache_migration"]["reason"] == (
        "kv_migration_execution_failed"
    )
    assert result["kv_cache_migration"]["request_state"] == "recompute"
    assert "copy failed" in result["kv_cache_migration"]["detail"]
    assert "migrate_kv_failed" in events
    assert "clear_kv_snapshot" in events


def test_engine_core_recommend_runtime_topology_uses_scheduler_load(monkeypatch):
    monkeypatch.setenv(
        "VLLM_PREBUILD_MODEL_PARALLEL_TOPOLOGIES",
        "tp=1,pp=4;tp=2,pp=2",
    )
    engine = EngineCore.__new__(EngineCore)
    engine.vllm_config = make_config(tp=4, pp=1, world_size=4)
    engine.scheduler = SimpleNamespace(get_request_counts=lambda: (3, 3))

    result = engine.recommend_runtime_topology()

    assert result["reason"] == "high_concurrency"
    assert result["workload"] == {
        "num_running_requests": 3,
        "num_waiting_requests": 3,
    }
    assert result["target"] == {
        "world_size": 4,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 4,
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "data_parallel_size": 1,
    }


def test_llm_engine_switch_runtime_topology_delegates_to_core_client():
    calls = []
    engine = LLMEngine.__new__(LLMEngine)

    def switch_runtime_topology(
        tp,
        pp,
        max_blocks_per_step=1,
        kv_migration_data_plane="cpu_staging",
    ):
        calls.append((tp, pp, max_blocks_per_step, kv_migration_data_plane))
        return {
            "target": {
                "tensor_parallel_size": tp,
                "pipeline_parallel_size": pp,
            }
        }

    engine.engine_core = SimpleNamespace(
        switch_runtime_topology=switch_runtime_topology
    )

    result = engine.switch_runtime_topology(
        tensor_parallel_size=1,
        pipeline_parallel_size=2,
        max_kv_migration_blocks_per_step=4,
    )

    assert calls == [(1, 2, 4, "cpu_staging")]
    assert result["target"]["pipeline_parallel_size"] == 2


def test_llm_engine_recommend_runtime_topology_delegates_to_core_client():
    engine = LLMEngine.__new__(LLMEngine)
    engine.engine_core = SimpleNamespace(
        recommend_runtime_topology=lambda: {"reason": "low_concurrency"}
    )

    result = engine.recommend_runtime_topology()

    assert result == {"reason": "low_concurrency"}


def test_engine_core_proc_input_queue_stops_after_pending_utility_future():
    proc = make_runtime_topology_engine_core_proc()
    events: list[str] = []
    pending: Future[str] = Future()

    def first_utility() -> Future[str]:
        events.append("first")
        return pending

    def second_utility() -> str:
        events.append("second")
        return "second"

    proc.first_utility = first_utility
    proc.second_utility = second_utility
    proc.input_queue.put(
        (EngineCoreRequestType.UTILITY, (0, 1, "first_utility", ()))
    )
    proc.input_queue.put(
        (EngineCoreRequestType.UTILITY, (0, 2, "second_utility", ()))
    )

    proc._process_input_queue()

    assert events == ["first"]
    assert proc.input_queue.qsize() == 1
    assert proc.output_queue.empty()

    pending.set_result("first-result")
    client_idx, output = proc.output_queue.get_nowait()
    assert client_idx == 0
    assert output.utility_output.call_id == 1
    assert output.utility_output.result.result == "first-result"


def test_engine_core_proc_input_queue_stops_after_idle_callback():
    proc = make_runtime_topology_engine_core_proc()
    events: list[str] = []

    def idle_callback(engine: EngineCoreProc) -> None:
        assert engine is proc
        events.append("idle")

    def queued_utility() -> str:
        events.append("queued")
        return "queued"

    proc.queued_utility = queued_utility
    proc._idle_state_callbacks.append(idle_callback)
    proc.input_queue.put(
        (EngineCoreRequestType.UTILITY, (0, 1, "queued_utility", ()))
    )

    proc._process_input_queue()

    assert events == ["idle"]
    assert proc.input_queue.qsize() == 1
    assert proc.output_queue.empty()


def test_sync_mp_client_switch_runtime_topology_sends_msgpack_safe_dict():
    sent = []
    client = SyncMPClient.__new__(SyncMPClient)
    client.call_utility = lambda method, payload: sent.append((method, payload)) or {
        "target": payload
    }

    result = client.switch_runtime_topology(
        tensor_parallel_size=1,
        pipeline_parallel_size=2,
        max_kv_migration_blocks_per_step=4,
    )

    assert sent == [
        (
            "switch_runtime_topology",
            {
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 2,
                "max_kv_migration_blocks_per_step": 4,
            },
        )
    ]
    assert result["target"]["pipeline_parallel_size"] == 2


def test_sync_mp_client_switch_runtime_topology_sends_kv_data_plane():
    sent = []
    client = SyncMPClient.__new__(SyncMPClient)
    client.call_utility = lambda method, payload: sent.append((method, payload)) or {
        "target": payload
    }

    result = client.switch_runtime_topology(
        tensor_parallel_size=1,
        pipeline_parallel_size=2,
        kv_migration_data_plane="p2p",
    )

    assert sent[0][0] == "switch_runtime_topology"
    assert sent[0][1]["kv_migration_data_plane"] == "p2p"
    assert result["target"]["kv_migration_data_plane"] == "p2p"


def test_sync_mp_client_recommend_runtime_topology_sends_utility_call():
    sent = []
    client = SyncMPClient.__new__(SyncMPClient)
    client.call_utility = lambda method: sent.append(method) or {
        "reason": "balanced_concurrency"
    }

    result = client.recommend_runtime_topology()

    assert sent == ["recommend_runtime_topology"]
    assert result["reason"] == "balanced_concurrency"


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
            max_kv_migration_blocks_per_step=4,
        )
    )

    assert sent == [
        (
            "switch_runtime_topology",
            {
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 2,
                "max_kv_migration_blocks_per_step": 4,
            },
        )
    ]
    assert result["target"]["pipeline_parallel_size"] == 2


def test_async_mp_client_switch_runtime_topology_sends_kv_data_plane():
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
            kv_migration_data_plane="p2p",
        )
    )

    assert sent[0][0] == "switch_runtime_topology"
    assert sent[0][1]["kv_migration_data_plane"] == "p2p"
    assert result["target"]["kv_migration_data_plane"] == "p2p"


def test_async_mp_client_recommend_runtime_topology_sends_utility_call():
    import asyncio

    sent = []
    client = AsyncMPClient.__new__(AsyncMPClient)

    async def call_utility_async(method):
        sent.append(method)
        return {"reason": "balanced_concurrency"}

    client.call_utility_async = call_utility_async

    result = asyncio.run(client.recommend_runtime_topology_async())

    assert sent == ["recommend_runtime_topology"]
    assert result["reason"] == "balanced_concurrency"
