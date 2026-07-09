# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.distributed.topology_cache import TopologyDescriptor
from vllm.v1.core.kv_cache_migration import (
    RuntimeKVMigrationCopyStats,
    RuntimeKVHeadPartition,
    RuntimeKVLayerPartition,
    RuntimeKVMigrationPlan,
    RuntimeKVMigrationPolicy,
    RuntimeKVSourceTensor,
)
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

    def __init__(self, snapshot=None, target=None) -> None:
        self.calls = []
        self.snapshot = snapshot
        self.target = target
        if target is not None:
            self._runtime_topology_kv_caches_by_layer = target

    def clear_runtime_state_for_topology_rebuild(
        self,
        *,
        clear_model: bool,
    ) -> None:
        self.calls.append(("clear_runtime", clear_model))

    def snapshot_runtime_kv_caches_for_topology_migration(self) -> None:
        self.calls.append("snapshot_kv")
        return self.snapshot

    def clear_runtime_kv_migration_snapshot(self) -> None:
        self.calls.append("clear_kv_snapshot")


def _migration_plan() -> RuntimeKVMigrationPlan:
    return RuntimeKVMigrationPlan(
        source_topology=TopologyDescriptor(
            world_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        target_topology=TopologyDescriptor(
            world_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        policy=RuntimeKVMigrationPolicy.MIGRATE,
        reason="capacity_available",
        pp_partitions=[
            RuntimeKVLayerPartition(pp_rank=0, layer_indices=range(0, 1)),
        ],
        tp_partitions=[
            RuntimeKVHeadPartition(tp_rank=0, head_indices=range(0, 1)),
        ],
        live_blocks=1,
        target_num_blocks=2,
        layer_names=("model.layers.0.self_attn",),
    )


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
    executor.snapshot_runtime_kv_caches_for_topology_migration()
    executor.rebuild_model_for_runtime_topology()
    executor.clear_runtime_kv_migration_snapshot()
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
        (
            "snapshot_runtime_kv_caches_for_topology_migration",
            None,
            (),
            None,
            False,
        ),
        ("rebuild_model_for_runtime_topology", None, (), None, False),
        ("clear_runtime_kv_migration_snapshot", None, (), None, False),
        ("clear_runtime_kv_state", None, (), None, False),
    ]


def test_executor_migrates_runtime_kv_cache_via_cpu_staging():
    executor = _fake_executor()
    plan = _migration_plan()
    block_mapping = {0: 0}
    source_tensor = torch.ones(2, 2, 2, 1, 1)

    def collective_rpc(method, timeout=None, args=(), kwargs=None, non_block=False):
        executor.calls.append((method, timeout, args, kwargs, non_block))
        if method == "export_runtime_kv_source_shard_for_migration":
            assert args == (plan,)
            assert kwargs == {
                "layer_names": ("model.layers.0.self_attn",),
                "block_ids": (0,),
                "head_indices": (0,),
            }
            return [
                {
                    "rank": 0,
                    "pp_rank": 0,
                    "tp_rank": 0,
                    "kv_caches": {
                        "model.layers.0.self_attn": source_tensor,
                    },
                }
            ]
        if method == "migrate_runtime_kv_cache_for_topology":
            assert kwargs is not None
            assert kwargs["plan"] is plan
            assert kwargs["block_mapping"] == block_mapping
            assert kwargs["max_blocks_per_step"] == 2
            assert kwargs["source_kv_caches"] == {
                (0, 0): {
                    "model.layers.0.self_attn": source_tensor,
                }
            }
            return [
                {
                    "migration_steps": 3,
                    "tensor_copies": 5,
                }
            ]
        return []

    executor.collective_rpc = collective_rpc

    result = executor.migrate_runtime_kv_cache_for_topology(
        plan=plan,
        block_mapping=block_mapping,
        max_blocks_per_step=2,
    )

    assert result == {
        "migration_steps": 3,
        "tensor_copies": 5,
        "source_shards": 1,
    }


def test_executor_streams_runtime_kv_source_one_layer_at_a_time():
    executor = _fake_executor()
    layer_names = tuple(f"model.layers.{i}.self_attn" for i in range(4))
    plan = RuntimeKVMigrationPlan(
        source_topology=TopologyDescriptor(
            world_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        target_topology=TopologyDescriptor(
            world_size=2,
            tensor_parallel_size=1,
            pipeline_parallel_size=2,
        ),
        policy=RuntimeKVMigrationPolicy.MIGRATE,
        reason="capacity_available",
        pp_partitions=[
            RuntimeKVLayerPartition(pp_rank=0, layer_indices=range(0, 2)),
            RuntimeKVLayerPartition(pp_rank=1, layer_indices=range(2, 4)),
        ],
        tp_partitions=[
            RuntimeKVHeadPartition(tp_rank=0, head_indices=range(0, 1)),
        ],
        live_blocks=1,
        target_num_blocks=2,
        layer_names=layer_names,
    )
    source_tensors = {
        layer_name: torch.full((2, 2, 2, 1, 1), fill_value=index)
        for index, layer_name in enumerate(layer_names)
    }
    export_layer_calls = []
    migrate_layer_calls = []
    migrate_pp_partition_calls = []
    migrate_layer_index_calls = []

    def collective_rpc(method, timeout=None, args=(), kwargs=None, non_block=False):
        executor.calls.append((method, timeout, args, kwargs, non_block))
        if method == "export_runtime_kv_source_shard_for_migration":
            assert args == (plan,)
            assert kwargs is not None
            exported_layers = tuple(kwargs["layer_names"])
            export_layer_calls.append(exported_layers)
            return [
                {
                    "rank": 0,
                    "pp_rank": 0,
                    "tp_rank": 0,
                    "kv_caches": {
                        layer_name: source_tensors[layer_name]
                        for layer_name in exported_layers
                    },
                }
            ]
        if method == "migrate_runtime_kv_cache_for_topology":
            assert kwargs is not None
            migrate_pp_partition_calls.append(
                tuple(
                    partition.pp_rank
                    for partition in kwargs["plan"].pp_partitions
                )
            )
            migrate_layer_index_calls.append(
                tuple(
                    tuple(partition.layer_indices)
                    for partition in kwargs["plan"].pp_partitions
                )
            )
            migrated_layers = tuple(kwargs["source_kv_caches"][(0, 0)])
            migrate_layer_calls.append(migrated_layers)
            return [
                {
                    "migration_steps": 1,
                    "tensor_copies": len(migrated_layers),
                }
            ]
        return []

    executor.collective_rpc = collective_rpc

    result = executor.migrate_runtime_kv_cache_for_topology(
        plan=plan,
        block_mapping={1: 1},
        max_blocks_per_step=1,
    )

    assert export_layer_calls == [
        (layer_names[0],),
        (layer_names[1],),
        (layer_names[2],),
        (layer_names[3],),
    ]
    assert migrate_layer_calls == export_layer_calls
    assert migrate_pp_partition_calls == [(0,), (0,), (1,), (1,)]
    assert migrate_layer_index_calls == [
        ((0,),),
        ((1,),),
        ((2,),),
        ((3,),),
    ]
    assert result == {
        "migration_steps": 4,
        "tensor_copies": 4,
        "source_shards": 1,
    }


def test_executor_streams_runtime_kv_source_blocks_within_each_layer():
    executor = _fake_executor()
    layer_name = "model.layers.0.self_attn"
    plan = _migration_plan()
    source_tensor = torch.arange(5 * 2 * 2, dtype=torch.float32).view(
        5, 2, 2, 1, 1
    )
    export_block_calls = []
    migrate_block_mappings = []
    migrate_live_blocks = []

    def collective_rpc(method, timeout=None, args=(), kwargs=None, non_block=False):
        executor.calls.append((method, timeout, args, kwargs, non_block))
        if method == "export_runtime_kv_source_shard_for_migration":
            assert args == (plan,)
            assert kwargs is not None
            assert kwargs["layer_names"] == (layer_name,)
            block_ids = tuple(kwargs["block_ids"])
            export_block_calls.append(block_ids)
            return [
                {
                    "rank": 0,
                    "pp_rank": 0,
                    "tp_rank": 0,
                    "kv_caches": {
                        layer_name: source_tensor[list(block_ids)].clone(),
                    },
                }
            ]
        if method == "migrate_runtime_kv_cache_for_topology":
            assert kwargs is not None
            migrate_block_mappings.append(kwargs["block_mapping"])
            migrate_live_blocks.append(kwargs["plan"].live_blocks)
            return [
                {
                    "migration_steps": 1,
                    "tensor_copies": 1,
                }
            ]
        return []

    executor.collective_rpc = collective_rpc

    result = executor.migrate_runtime_kv_cache_for_topology(
        plan=plan,
        block_mapping={1: 5, 3: 6, 4: 7},
        max_blocks_per_step=2,
    )

    assert export_block_calls == [(1, 3), (4,)]
    assert migrate_block_mappings == [{0: 5, 1: 6}, {0: 7}]
    assert migrate_live_blocks == [2, 1]
    assert result == {
        "migration_steps": 2,
        "tensor_copies": 2,
        "source_shards": 1,
    }


def test_executor_streams_runtime_kv_source_one_target_head_partition_at_a_time():
    executor = _fake_executor()
    layer_name = "model.layers.0.self_attn"
    plan = RuntimeKVMigrationPlan(
        source_topology=TopologyDescriptor(
            world_size=2,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
        ),
        target_topology=TopologyDescriptor(
            world_size=2,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
        ),
        policy=RuntimeKVMigrationPolicy.MIGRATE,
        reason="capacity_available",
        pp_partitions=[
            RuntimeKVLayerPartition(pp_rank=0, layer_indices=range(0, 1)),
        ],
        tp_partitions=[
            RuntimeKVHeadPartition(tp_rank=0, head_indices=range(0, 2)),
            RuntimeKVHeadPartition(tp_rank=1, head_indices=range(2, 4)),
        ],
        live_blocks=1,
        target_num_blocks=4,
        layer_names=(layer_name,),
        global_num_kv_heads=4,
    )
    source_tensor = torch.ones(1, 2, 2, 2, 1)
    export_head_calls = []
    migrate_tp_partition_calls = []

    def collective_rpc(method, timeout=None, args=(), kwargs=None, non_block=False):
        executor.calls.append((method, timeout, args, kwargs, non_block))
        if method == "export_runtime_kv_source_shard_for_migration":
            assert args == (plan,)
            assert kwargs is not None
            head_indices = tuple(kwargs["head_indices"])
            export_head_calls.append(head_indices)
            return [
                {
                    "rank": head_indices[0] // 2,
                    "pp_rank": 0,
                    "tp_rank": head_indices[0] // 2,
                    "kv_caches": {
                        layer_name: RuntimeKVSourceTensor(
                            tensor=source_tensor,
                            head_indices=head_indices,
                        )
                    },
                }
            ]
        if method == "migrate_runtime_kv_cache_for_topology":
            assert kwargs is not None
            migrate_tp_partition_calls.append(
                tuple(
                    (partition.tp_rank, tuple(partition.head_indices))
                    for partition in kwargs["plan"].tp_partitions
                )
            )
            return [
                {
                    "migration_steps": 1,
                    "tensor_copies": 2,
                }
            ]
        return []

    executor.collective_rpc = collective_rpc

    result = executor.migrate_runtime_kv_cache_for_topology(
        plan=plan,
        block_mapping={1: 1},
        max_blocks_per_step=1,
    )

    assert export_head_calls == [(0, 1), (2, 3)]
    assert migrate_tp_partition_calls == [
        ((0, (0, 1)),),
        ((1, (2, 3)),),
    ]
    assert result == {
        "migration_steps": 2,
        "tensor_copies": 4,
        "source_shards": 2,
    }


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


def test_gpu_worker_snapshots_runtime_kv_before_topology_migration():
    worker = Worker.__new__(Worker)
    runner = FakeModelRunner(snapshot={"layer": object()})
    worker.model_runner = runner

    worker.snapshot_runtime_kv_caches_for_topology_migration()
    worker.clear_runtime_kv_migration_snapshot()

    assert runner.calls == ["snapshot_kv", "clear_kv_snapshot"]
    assert worker._runtime_topology_source_kv_caches is None


def test_gpu_worker_preserves_runtime_kv_snapshot_across_model_rebuild():
    worker = Worker.__new__(Worker)
    snapshot = {"model.layers.0.self_attn": object()}
    old_runner = FakeModelRunner(snapshot=snapshot)
    new_runner = FakeModelRunner()
    worker.model_runner = old_runner
    calls = []

    def init_model_runner():
        calls.append("init_model_runner")
        worker.model_runner = new_runner

    def load_model():
        calls.append("load_model")

    worker._init_model_runner = init_model_runner
    worker.load_model = load_model

    worker.snapshot_runtime_kv_caches_for_topology_migration()
    worker.rebuild_model_for_runtime_topology()

    assert worker._runtime_topology_source_kv_caches == snapshot
    assert old_runner.calls == ["snapshot_kv", ("clear_runtime", True)]
    assert calls == ["init_model_runner", "load_model"]


def test_gpu_worker_exports_runtime_kv_source_shard_for_migration():
    worker = Worker.__new__(Worker)
    worker.rank = 0
    source_tensor = torch.ones(2, 2, 2, 1, 1)
    worker._runtime_topology_source_kv_caches = {
        "model.layers.0.self_attn": source_tensor
    }
    plan = _migration_plan()

    shard = worker.export_runtime_kv_source_shard_for_migration(plan)

    assert shard is not None
    assert shard["rank"] == 0
    assert shard["pp_rank"] == 0
    assert shard["tp_rank"] == 0
    exported_tensor = shard["kv_caches"]["model.layers.0.self_attn"]
    assert exported_tensor.device.type == "cpu"
    assert exported_tensor is not source_tensor
    torch.testing.assert_close(exported_tensor, source_tensor.cpu())


def test_gpu_worker_exports_only_requested_runtime_kv_source_layers():
    worker = Worker.__new__(Worker)
    worker.rank = 0
    layer0 = torch.ones(2, 2, 2, 1, 1)
    layer1 = torch.full((2, 2, 2, 1, 1), fill_value=2)
    worker._runtime_topology_source_kv_caches = {
        "model.layers.0.self_attn": layer0,
        "model.layers.1.self_attn": layer1,
    }
    plan = _migration_plan()

    shard = worker.export_runtime_kv_source_shard_for_migration(
        plan,
        layer_names=("model.layers.1.self_attn",),
    )

    assert shard is not None
    assert tuple(shard["kv_caches"]) == ("model.layers.1.self_attn",)
    exported_tensor = shard["kv_caches"]["model.layers.1.self_attn"]
    assert exported_tensor is not layer1
    torch.testing.assert_close(exported_tensor, layer1.cpu())


def test_gpu_worker_exports_only_requested_runtime_kv_source_blocks():
    worker = Worker.__new__(Worker)
    worker.rank = 0
    source_tensor = torch.arange(4 * 2 * 2, dtype=torch.float32).view(
        4, 2, 2, 1, 1
    )
    worker._runtime_topology_source_kv_caches = {
        "model.layers.0.self_attn": source_tensor,
    }
    plan = _migration_plan()

    shard = worker.export_runtime_kv_source_shard_for_migration(
        plan,
        layer_names=("model.layers.0.self_attn",),
        block_ids=(3, 1),
    )

    assert shard is not None
    exported_tensor = shard["kv_caches"]["model.layers.0.self_attn"]
    assert exported_tensor.shape[0] == 2
    assert exported_tensor is not source_tensor
    torch.testing.assert_close(exported_tensor[0], source_tensor[3].cpu())
    torch.testing.assert_close(exported_tensor[1], source_tensor[1].cpu())


def test_gpu_worker_exports_only_intersecting_runtime_kv_source_heads():
    worker = Worker.__new__(Worker)
    worker.rank = 1
    source_tensor = torch.arange(4 * 2 * 2 * 2, dtype=torch.float32).view(
        4, 2, 2, 2, 1
    )
    worker._runtime_topology_source_kv_caches = {
        "model.layers.0.self_attn": source_tensor,
    }
    plan = RuntimeKVMigrationPlan(
        source_topology=TopologyDescriptor(
            world_size=2,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
        ),
        target_topology=TopologyDescriptor(
            world_size=2,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
        ),
        policy=RuntimeKVMigrationPolicy.MIGRATE,
        reason="capacity_available",
        pp_partitions=[
            RuntimeKVLayerPartition(pp_rank=0, layer_indices=range(0, 1)),
        ],
        tp_partitions=[
            RuntimeKVHeadPartition(tp_rank=0, head_indices=range(0, 2)),
            RuntimeKVHeadPartition(tp_rank=1, head_indices=range(2, 4)),
        ],
        live_blocks=1,
        target_num_blocks=4,
        layer_names=("model.layers.0.self_attn",),
        global_num_kv_heads=4,
    )

    shard = worker.export_runtime_kv_source_shard_for_migration(
        plan,
        layer_names=("model.layers.0.self_attn",),
        block_ids=(3, 1),
        head_indices=(2,),
    )

    assert shard is not None
    exported = shard["kv_caches"]["model.layers.0.self_attn"]
    assert isinstance(exported, RuntimeKVSourceTensor)
    assert exported.head_indices == (2,)
    assert exported.tensor.shape == (2, 2, 2, 1, 1)
    torch.testing.assert_close(exported.tensor[0, :, :, 0, :], source_tensor[3, :, :, 0, :])
    torch.testing.assert_close(exported.tensor[1, :, :, 0, :], source_tensor[1, :, :, 0, :])


def test_gpu_worker_executes_runtime_kv_migration_on_target_shard(monkeypatch):
    worker = Worker.__new__(Worker)
    worker.rank = 0
    target = {"model.layers.0.self_attn": torch.zeros(2, 2, 2, 1, 1)}
    worker.model_runner = FakeModelRunner(target=target)
    source = {(0, 0): {"model.layers.0.self_attn": torch.ones(2, 2, 2, 1, 1)}}
    plan = _migration_plan()
    calls = []

    def migrate(**kwargs):
        calls.append(kwargs)
        return RuntimeKVMigrationCopyStats(migration_steps=1, tensor_copies=1)

    monkeypatch.setattr(
        "vllm.v1.worker.gpu_worker.migrate_runtime_kv_cache_shard",
        migrate,
    )

    result = worker.migrate_runtime_kv_cache_for_topology(
        plan=plan,
        source_kv_caches=source,
        block_mapping={0: 0},
        max_blocks_per_step=1,
    )

    assert result == {"migration_steps": 1, "tensor_copies": 1}
    assert calls[0]["target_kv_caches"] is target


def test_gpu_worker_skips_runtime_kv_migration_when_batch_plan_excludes_pp(
    monkeypatch,
):
    worker = Worker.__new__(Worker)
    worker.rank = 1
    target = {"model.layers.0.self_attn": torch.zeros(2, 2, 2, 1, 1)}
    worker.model_runner = FakeModelRunner(target=target)
    plan = RuntimeKVMigrationPlan(
        source_topology=TopologyDescriptor(
            world_size=2,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
        ),
        target_topology=TopologyDescriptor(
            world_size=2,
            tensor_parallel_size=1,
            pipeline_parallel_size=2,
        ),
        policy=RuntimeKVMigrationPolicy.MIGRATE,
        reason="capacity_available",
        pp_partitions=[
            RuntimeKVLayerPartition(pp_rank=0, layer_indices=range(0, 1)),
        ],
        tp_partitions=[
            RuntimeKVHeadPartition(tp_rank=0, head_indices=range(0, 1)),
        ],
        live_blocks=1,
        target_num_blocks=2,
        layer_names=("model.layers.0.self_attn",),
    )

    def fail_if_called(**kwargs):
        raise AssertionError("excluded target PP shard should not migrate")

    monkeypatch.setattr(
        "vllm.v1.worker.gpu_worker.migrate_runtime_kv_cache_shard",
        fail_if_called,
    )

    result = worker.migrate_runtime_kv_cache_for_topology(
        plan=plan,
        source_kv_caches={
            (0, 0): {
                "model.layers.0.self_attn": torch.ones(2, 2, 2, 1, 1),
            }
        },
        block_mapping={0: 0},
        max_blocks_per_step=1,
    )

    assert result == {"migration_steps": 0, "tensor_copies": 0}


def test_gpu_worker_skips_runtime_kv_migration_when_batch_plan_excludes_tp(
    monkeypatch,
):
    worker = Worker.__new__(Worker)
    worker.rank = 1
    target = {"model.layers.0.self_attn": torch.zeros(2, 2, 2, 1, 1)}
    worker.model_runner = FakeModelRunner(target=target)
    plan = RuntimeKVMigrationPlan(
        source_topology=TopologyDescriptor(
            world_size=2,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
        ),
        target_topology=TopologyDescriptor(
            world_size=2,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
        ),
        policy=RuntimeKVMigrationPolicy.MIGRATE,
        reason="capacity_available",
        pp_partitions=[
            RuntimeKVLayerPartition(pp_rank=0, layer_indices=range(0, 1)),
        ],
        tp_partitions=[
            RuntimeKVHeadPartition(tp_rank=0, head_indices=range(0, 1)),
        ],
        live_blocks=1,
        target_num_blocks=2,
        layer_names=("model.layers.0.self_attn",),
        global_num_kv_heads=2,
    )

    def fail_if_called(**kwargs):
        raise AssertionError("excluded target TP shard should not migrate")

    monkeypatch.setattr(
        "vllm.v1.worker.gpu_worker.migrate_runtime_kv_cache_shard",
        fail_if_called,
    )

    result = worker.migrate_runtime_kv_cache_for_topology(
        plan=plan,
        source_kv_caches={
            (0, 0): {
                "model.layers.0.self_attn": torch.ones(2, 2, 2, 1, 1),
            }
        },
        block_mapping={0: 0},
        max_blocks_per_step=1,
    )

    assert result == {"migration_steps": 0, "tensor_copies": 0}


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
