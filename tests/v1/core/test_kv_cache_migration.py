# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.distributed.topology_cache import TopologyDescriptor
from vllm.v1.core.kv_cache_migration import (
    RuntimeKVHeadPartition,
    RuntimeKVLayerPartition,
    RuntimeKVMigrationPlan,
    RuntimeKVMigrationPolicy,
    RuntimeKVSourceTensor,
    build_global_runtime_kv_cache_config,
    build_runtime_kv_block_mapping,
    build_runtime_kv_migration_plan,
    get_runtime_topology_kv_rank,
    migrate_runtime_kv_cache,
    migrate_runtime_kv_cache_shard,
    iter_layerwise_kv_migration_steps,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)


def _kv_config(*, num_blocks: int = 8) -> KVCacheConfig:
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=4,
        head_size=64,
        dtype=torch.float16,
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=[f"model.layers.{i}.self_attn" for i in range(4)],
                kv_cache_spec=spec,
            )
        ],
    )


def test_global_runtime_kv_cache_config_merges_pp_layers_and_global_heads():
    local_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.float16,
    )
    worker_configs = [
        KVCacheConfig(
            num_blocks=8,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=[
                        "model.layers.0.self_attn",
                        "model.layers.1.self_attn",
                    ],
                    kv_cache_spec=local_spec,
                )
            ],
        ),
        KVCacheConfig(
            num_blocks=8,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=[
                        "model.layers.2.self_attn",
                        "model.layers.3.self_attn",
                    ],
                    kv_cache_spec=local_spec,
                )
            ],
        ),
    ]

    global_config = build_global_runtime_kv_cache_config(
        worker_configs,
        target_topology=TopologyDescriptor(
            world_size=4,
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
        ),
    )
    plan = build_runtime_kv_migration_plan(
        source_topology=TopologyDescriptor(
            world_size=4,
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
        ),
        target_topology=TopologyDescriptor(
            world_size=4,
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
        ),
        kv_cache_config=global_config,
        live_blocks=2,
    )

    assert global_config.kv_cache_groups[0].layer_names == [
        "model.layers.0.self_attn",
        "model.layers.1.self_attn",
        "model.layers.2.self_attn",
        "model.layers.3.self_attn",
    ]
    assert global_config.kv_cache_groups[0].kv_cache_spec.num_kv_heads == 4
    assert [partition.layer_indices for partition in plan.pp_partitions] == [
        range(0, 2),
        range(2, 4),
    ]
    assert [partition.head_indices for partition in plan.tp_partitions] == [
        range(0, 2),
        range(2, 4),
    ]


def _make_kv_tensor(num_blocks: int, local_heads: int) -> torch.Tensor:
    return torch.full((num_blocks, 2, 2, local_heads, 3), -1, dtype=torch.int64)


def _fill_layer_cache(
    tensor: torch.Tensor,
    *,
    layer_index: int,
    global_heads: range,
) -> None:
    for block_id in range(tensor.shape[0]):
        for local_head, global_head in enumerate(global_heads):
            tensor[block_id, :, :, local_head, :] = (
                layer_index * 10000
                + global_head * 1000
                + block_id * 100
                + torch.arange(12, dtype=torch.int64).view(2, 2, 3)
            )


def _source_tp4_pp1_kv_caches() -> dict[tuple[int, int], dict[str, torch.Tensor]]:
    source: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
    for tp_rank in range(4):
        shard: dict[str, torch.Tensor] = {}
        for layer_index in range(4):
            tensor = _make_kv_tensor(num_blocks=8, local_heads=1)
            _fill_layer_cache(
                tensor,
                layer_index=layer_index,
                global_heads=range(tp_rank, tp_rank + 1),
            )
            shard[f"model.layers.{layer_index}.self_attn"] = tensor
        source[(0, tp_rank)] = shard
    return source


def _target_tp2_pp2_kv_caches() -> dict[tuple[int, int], dict[str, torch.Tensor]]:
    target: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
    for pp_rank, layer_indices in enumerate((range(0, 2), range(2, 4))):
        for tp_rank in range(2):
            target[(pp_rank, tp_rank)] = {
                f"model.layers.{layer_index}.self_attn": _make_kv_tensor(
                    num_blocks=8,
                    local_heads=2,
                )
                for layer_index in layer_indices
            }
    return target


def test_runtime_kv_migration_plan_covers_pp_layers_and_tp_heads():
    plan = build_runtime_kv_migration_plan(
        source_topology=TopologyDescriptor(
            world_size=4,
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
        ),
        target_topology=TopologyDescriptor(
            world_size=4,
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
        ),
        kv_cache_config=_kv_config(),
        live_blocks=4,
    )

    assert plan.policy == RuntimeKVMigrationPolicy.MIGRATE
    assert [partition.layer_indices for partition in plan.pp_partitions] == [
        range(0, 2),
        range(2, 4),
    ]
    assert [partition.head_indices for partition in plan.tp_partitions] == [
        range(0, 2),
        range(2, 4),
    ]


def test_runtime_topology_kv_rank_maps_global_rank_to_pp_and_tp():
    descriptor = TopologyDescriptor(
        world_size=4,
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
    )

    assert get_runtime_topology_kv_rank(descriptor, rank=0) == (0, 0)
    assert get_runtime_topology_kv_rank(descriptor, rank=1) == (0, 1)
    assert get_runtime_topology_kv_rank(descriptor, rank=2) == (1, 0)
    assert get_runtime_topology_kv_rank(descriptor, rank=3) == (1, 1)


def test_runtime_kv_migration_plan_recomputes_when_capacity_is_insufficient():
    plan = build_runtime_kv_migration_plan(
        source_topology=TopologyDescriptor(
            world_size=4,
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
        ),
        target_topology=TopologyDescriptor(
            world_size=4,
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
        ),
        kv_cache_config=_kv_config(num_blocks=2),
        live_blocks=3,
    )

    assert plan.policy == RuntimeKVMigrationPolicy.RECOMPUTE
    assert plan.reason == "insufficient_target_kv_capacity"


def test_runtime_kv_migration_plan_rejects_uneven_kv_heads():
    with pytest.raises(ValueError, match="KV heads"):
        build_runtime_kv_migration_plan(
            source_topology=TopologyDescriptor(
                world_size=3,
                tensor_parallel_size=3,
                pipeline_parallel_size=1,
            ),
            target_topology=TopologyDescriptor(
                world_size=3,
                tensor_parallel_size=3,
                pipeline_parallel_size=1,
            ),
            kv_cache_config=KVCacheConfig(
                num_blocks=8,
                kv_cache_tensors=[],
                kv_cache_groups=[
                    KVCacheGroupSpec(
                        layer_names=["model.layers.0.self_attn"],
                        kv_cache_spec=FullAttentionSpec(
                            block_size=16,
                            num_kv_heads=4,
                            head_size=64,
                            dtype=torch.float16,
                        ),
                    )
                ],
            ),
            live_blocks=1,
        )


def test_layerwise_kv_migration_steps_chunk_blocks_to_bound_peak_memory():
    plan = build_runtime_kv_migration_plan(
        source_topology=TopologyDescriptor(
            world_size=4,
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
        ),
        target_topology=TopologyDescriptor(
            world_size=4,
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
        ),
        kv_cache_config=_kv_config(),
        live_blocks=5,
    )

    steps = list(
        iter_layerwise_kv_migration_steps(
            plan,
            block_ids=[10, 11, 12, 13, 14],
            max_blocks_per_step=2,
        )
    )

    assert steps[0].pp_rank == 0
    assert steps[0].tp_rank == 0
    assert steps[0].layer_indices == range(0, 2)
    assert steps[0].head_indices == range(0, 2)
    assert steps[0].block_ids == (10, 11)
    assert max(len(step.block_ids) for step in steps) == 2
    assert {
        (step.pp_rank, step.tp_rank, step.layer_indices, step.head_indices)
        for step in steps
    } == {
        (0, 0, range(0, 2), range(0, 2)),
        (0, 1, range(0, 2), range(2, 4)),
        (1, 0, range(2, 4), range(0, 2)),
        (1, 1, range(2, 4), range(2, 4)),
    }


def test_runtime_kv_block_mapping_preserves_physical_ids_when_possible():
    mapping = build_runtime_kv_block_mapping(
        request_block_ids={
            "req-0": [2, 4, 6],
            "req-1": [2, 6, 7],
        },
        target_num_blocks=8,
    )

    assert mapping.live_blocks == 4
    assert mapping.block_mapping == {
        2: 2,
        4: 4,
        6: 6,
        7: 7,
    }


def test_runtime_kv_block_mapping_compacts_ids_when_preserve_is_not_possible():
    mapping = build_runtime_kv_block_mapping(
        request_block_ids={
            "req-0": [10, 12],
            "req-1": [10, 14],
        },
        target_num_blocks=4,
    )

    assert mapping.live_blocks == 3
    assert mapping.block_mapping == {
        10: 1,
        12: 2,
        14: 3,
    }


def test_runtime_kv_block_mapping_ignores_null_block_zero():
    mapping = build_runtime_kv_block_mapping(
        request_block_ids={
            "req-0": [0, 2, 0, 4],
            "req-1": [0, 2],
        },
        target_num_blocks=3,
    )

    assert mapping.live_blocks == 2
    assert mapping.block_mapping == {
        2: 1,
        4: 2,
    }


def test_runtime_kv_block_mapping_rejects_insufficient_target_capacity():
    with pytest.raises(ValueError, match="target KV capacity"):
        build_runtime_kv_block_mapping(
            request_block_ids={
                "req-0": [10, 12],
                "req-1": [10, 14],
            },
            target_num_blocks=2,
        )


def test_runtime_kv_block_mapping_reserves_null_block_zero():
    with pytest.raises(ValueError, match="target KV capacity"):
        build_runtime_kv_block_mapping(
            request_block_ids={
                "req-0": [10, 12],
                "req-1": [14],
            },
            target_num_blocks=3,
        )


def test_migrate_runtime_kv_cache_copies_pp_layers_tp_heads_and_remapped_blocks():
    plan = build_runtime_kv_migration_plan(
        source_topology=TopologyDescriptor(
            world_size=4,
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
        ),
        target_topology=TopologyDescriptor(
            world_size=4,
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
        ),
        kv_cache_config=_kv_config(),
        live_blocks=2,
    )
    source = _source_tp4_pp1_kv_caches()
    target = _target_tp2_pp2_kv_caches()

    stats = migrate_runtime_kv_cache(
        plan,
        source_kv_caches=source,
        target_kv_caches=target,
        block_mapping={1: 5, 3: 6},
        max_blocks_per_step=1,
    )

    # Target pp0/tp0 owns layers 0-1 and global heads 0-1.
    torch.testing.assert_close(
        target[(0, 0)]["model.layers.0.self_attn"][5, :, :, 0, :],
        source[(0, 0)]["model.layers.0.self_attn"][1, :, :, 0, :],
    )
    torch.testing.assert_close(
        target[(0, 0)]["model.layers.1.self_attn"][6, :, :, 1, :],
        source[(0, 1)]["model.layers.1.self_attn"][3, :, :, 0, :],
    )

    # Target pp1/tp1 owns layers 2-3 and global heads 2-3.
    torch.testing.assert_close(
        target[(1, 1)]["model.layers.2.self_attn"][5, :, :, 0, :],
        source[(0, 2)]["model.layers.2.self_attn"][1, :, :, 0, :],
    )
    torch.testing.assert_close(
        target[(1, 1)]["model.layers.3.self_attn"][6, :, :, 1, :],
        source[(0, 3)]["model.layers.3.self_attn"][3, :, :, 0, :],
    )

    assert torch.all(target[(0, 0)]["model.layers.0.self_attn"][0] == -1)
    assert stats.migration_steps == 8
    assert stats.tensor_copies == 32


def test_migrate_runtime_kv_cache_shard_copies_only_requested_target_shard():
    plan = build_runtime_kv_migration_plan(
        source_topology=TopologyDescriptor(
            world_size=4,
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
        ),
        target_topology=TopologyDescriptor(
            world_size=4,
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
        ),
        kv_cache_config=_kv_config(),
        live_blocks=2,
    )
    source = _source_tp4_pp1_kv_caches()
    target_shard = _target_tp2_pp2_kv_caches()[(1, 0)]

    # This target shard needs only source tp ranks 0 and 1 for layers 2 and 3.
    # The single-shard path must not require unrelated source shards or target
    # shards that belong to other workers.
    del source[(0, 2)]
    del source[(0, 3)]

    stats = migrate_runtime_kv_cache_shard(
        plan,
        source_kv_caches=source,
        target_kv_caches=target_shard,
        target_pp_rank=1,
        target_tp_rank=0,
        block_mapping={1: 5, 3: 6},
        max_blocks_per_step=1,
    )

    torch.testing.assert_close(
        target_shard["model.layers.2.self_attn"][5, :, :, 0, :],
        source[(0, 0)]["model.layers.2.self_attn"][1, :, :, 0, :],
    )
    torch.testing.assert_close(
        target_shard["model.layers.3.self_attn"][6, :, :, 1, :],
        source[(0, 1)]["model.layers.3.self_attn"][3, :, :, 0, :],
    )
    assert torch.all(target_shard["model.layers.2.self_attn"][0] == -1)
    assert stats.migration_steps == 2
    assert stats.tensor_copies == 8


def test_migrate_runtime_kv_cache_shard_accepts_sliced_source_heads():
    plan = RuntimeKVMigrationPlan(
        source_topology=TopologyDescriptor(
            world_size=4,
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
        ),
        target_topology=TopologyDescriptor(
            world_size=4,
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
        ),
        policy=RuntimeKVMigrationPolicy.MIGRATE,
        reason="capacity_available",
        pp_partitions=[
            RuntimeKVLayerPartition(pp_rank=0, layer_indices=range(0, 1)),
        ],
        tp_partitions=[
            RuntimeKVHeadPartition(tp_rank=2, head_indices=range(2, 3)),
        ],
        live_blocks=1,
        target_num_blocks=8,
        layer_names=("model.layers.0.self_attn",),
        global_num_kv_heads=4,
    )
    source_full = _make_kv_tensor(num_blocks=8, local_heads=2)
    _fill_layer_cache(
        source_full,
        layer_index=0,
        global_heads=range(2, 4),
    )
    source = {
        (0, 1): {
            "model.layers.0.self_attn": RuntimeKVSourceTensor(
                tensor=source_full[:, :, :, 0:1, :].clone(),
                head_indices=(2,),
            )
        }
    }
    target = {
        "model.layers.0.self_attn": _make_kv_tensor(
            num_blocks=8,
            local_heads=1,
        )
    }

    stats = migrate_runtime_kv_cache_shard(
        plan,
        source_kv_caches=source,
        target_kv_caches=target,
        target_pp_rank=0,
        target_tp_rank=2,
        block_mapping={1: 5},
        max_blocks_per_step=1,
    )

    torch.testing.assert_close(
        target["model.layers.0.self_attn"][5, :, :, 0, :],
        source_full[1, :, :, 0, :],
    )
    assert stats.migration_steps == 1
    assert stats.tensor_copies == 1


def test_migrate_runtime_kv_cache_refuses_recompute_plan():
    plan = build_runtime_kv_migration_plan(
        source_topology=TopologyDescriptor(
            world_size=4,
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
        ),
        target_topology=TopologyDescriptor(
            world_size=4,
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
        ),
        kv_cache_config=_kv_config(num_blocks=1),
        live_blocks=2,
    )

    with pytest.raises(ValueError, match="recompute"):
        migrate_runtime_kv_cache(
            plan,
            source_kv_caches={},
            target_kv_caches={},
            block_mapping={0: 0, 1: 1},
            max_blocks_per_step=1,
        )


def test_migrate_runtime_kv_cache_rejects_duplicate_target_blocks():
    plan = build_runtime_kv_migration_plan(
        source_topology=TopologyDescriptor(
            world_size=4,
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
        ),
        target_topology=TopologyDescriptor(
            world_size=4,
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
        ),
        kv_cache_config=_kv_config(),
        live_blocks=2,
    )

    with pytest.raises(ValueError, match="duplicate target block"):
        migrate_runtime_kv_cache(
            plan,
            source_kv_caches=_source_tp4_pp1_kv_caches(),
            target_kv_caches=_target_tp2_pp2_kv_caches(),
            block_mapping={0: 4, 1: 4},
            max_blocks_per_step=1,
        )


def test_migrate_runtime_kv_cache_rejects_target_null_block():
    plan = build_runtime_kv_migration_plan(
        source_topology=TopologyDescriptor(
            world_size=4,
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
        ),
        target_topology=TopologyDescriptor(
            world_size=4,
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
        ),
        kv_cache_config=_kv_config(),
        live_blocks=1,
    )

    with pytest.raises(ValueError, match="target block id.*null block 0"):
        migrate_runtime_kv_cache(
            plan,
            source_kv_caches=_source_tp4_pp1_kv_caches(),
            target_kv_caches=_target_tp2_pp2_kv_caches(),
            block_mapping={1: 0},
            max_blocks_per_step=1,
        )


def test_migrate_runtime_kv_cache_rejects_nonstandard_attention_layout():
    plan = build_runtime_kv_migration_plan(
        source_topology=TopologyDescriptor(
            world_size=4,
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
        ),
        target_topology=TopologyDescriptor(
            world_size=4,
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
        ),
        kv_cache_config=_kv_config(),
        live_blocks=1,
    )
    source = _source_tp4_pp1_kv_caches()
    target = _target_tp2_pp2_kv_caches()
    source[(0, 0)]["model.layers.0.self_attn"] = torch.zeros(8, 2, 2, 1)

    with pytest.raises(ValueError, match="standard attention KV cache layout"):
        migrate_runtime_kv_cache(
            plan,
            source_kv_caches=source,
            target_kv_caches=target,
            block_mapping={1: 1},
            max_blocks_per_step=1,
        )
