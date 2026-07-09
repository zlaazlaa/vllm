# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime model-parallel topology switch validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import vllm.envs as envs
from vllm.distributed.topology_cache import (
    TopologyDescriptor,
    parse_topology_descriptors,
)


@dataclass(frozen=True)
class RuntimeTopologySwitchRequest:
    tensor_parallel_size: int
    pipeline_parallel_size: int


@dataclass(frozen=True)
class RuntimeTopologySwitchPlan:
    previous_topology: TopologyDescriptor
    target_topology: TopologyDescriptor


@dataclass(frozen=True)
class RuntimeTopologyWorkload:
    num_running_requests: int
    num_waiting_requests: int

    def __post_init__(self) -> None:
        if self.num_running_requests < 0:
            raise ValueError("num_running_requests must be >= 0")
        if self.num_waiting_requests < 0:
            raise ValueError("num_waiting_requests must be >= 0")

    @property
    def total_requests(self) -> int:
        return self.num_running_requests + self.num_waiting_requests


@dataclass(frozen=True)
class RuntimeTopologyRecommendation:
    previous_topology: TopologyDescriptor
    target_topology: TopologyDescriptor
    workload: RuntimeTopologyWorkload
    reason: str


def topology_descriptor_to_dict(
    descriptor: TopologyDescriptor,
) -> dict[str, int]:
    return {
        "world_size": descriptor.world_size,
        "tensor_parallel_size": descriptor.tensor_parallel_size,
        "pipeline_parallel_size": descriptor.pipeline_parallel_size,
        "prefill_context_parallel_size": descriptor.prefill_context_parallel_size,
        "decode_context_parallel_size": descriptor.decode_context_parallel_size,
        "data_parallel_size": descriptor.data_parallel_size,
    }


def topology_recommendation_to_dict(
    recommendation: RuntimeTopologyRecommendation,
) -> dict[str, Any]:
    return {
        "previous": topology_descriptor_to_dict(
            recommendation.previous_topology
        ),
        "target": topology_descriptor_to_dict(recommendation.target_topology),
        "workload": {
            "num_running_requests": (
                recommendation.workload.num_running_requests
            ),
            "num_waiting_requests": (
                recommendation.workload.num_waiting_requests
            ),
        },
        "reason": recommendation.reason,
    }


def apply_runtime_topology_to_config(
    vllm_config: Any,
    descriptor: TopologyDescriptor,
) -> None:
    parallel_config = vllm_config.parallel_config
    parallel_config.world_size = descriptor.world_size
    parallel_config.tensor_parallel_size = descriptor.tensor_parallel_size
    parallel_config.pipeline_parallel_size = descriptor.pipeline_parallel_size
    parallel_config.prefill_context_parallel_size = (
        descriptor.prefill_context_parallel_size
    )
    parallel_config.decode_context_parallel_size = (
        descriptor.decode_context_parallel_size
    )
    parallel_config.data_parallel_size = descriptor.data_parallel_size


def _getattr_nested(obj: Any, path: str, default: Any = None) -> Any:
    current = obj
    for name in path.split("."):
        current = getattr(current, name, default)
        if current is default:
            return default
    return current


def _current_topology(vllm_config: Any) -> TopologyDescriptor:
    parallel_config = vllm_config.parallel_config
    return TopologyDescriptor(
        world_size=parallel_config.world_size,
        tensor_parallel_size=parallel_config.tensor_parallel_size,
        pipeline_parallel_size=parallel_config.pipeline_parallel_size,
        prefill_context_parallel_size=parallel_config.prefill_context_parallel_size,
        decode_context_parallel_size=(
            parallel_config.decode_context_parallel_size or 1
        ),
        data_parallel_size=parallel_config.data_parallel_size,
    )


def _target_topology(
    vllm_config: Any,
    request: RuntimeTopologySwitchRequest,
) -> TopologyDescriptor:
    parallel_config = vllm_config.parallel_config
    return TopologyDescriptor(
        world_size=parallel_config.world_size,
        tensor_parallel_size=request.tensor_parallel_size,
        pipeline_parallel_size=request.pipeline_parallel_size,
        prefill_context_parallel_size=parallel_config.prefill_context_parallel_size,
        decode_context_parallel_size=(
            parallel_config.decode_context_parallel_size or 1
        ),
        data_parallel_size=parallel_config.data_parallel_size,
    )


def collect_runtime_topology_candidates(
    vllm_config: Any,
) -> list[TopologyDescriptor]:
    current = _current_topology(vllm_config)
    prebuilt = {current.key: current}
    topology_spec = envs.VLLM_PREBUILD_MODEL_PARALLEL_TOPOLOGIES
    if not topology_spec:
        return list(prebuilt.values())

    parallel_config = vllm_config.parallel_config
    for descriptor in parse_topology_descriptors(
        topology_spec,
        world_size=parallel_config.world_size,
        data_parallel_size=parallel_config.data_parallel_size,
        prefill_context_parallel_size=(
            parallel_config.prefill_context_parallel_size
        ),
        decode_context_parallel_size=(
            parallel_config.decode_context_parallel_size or 1
        ),
    ):
        prebuilt[descriptor.key] = descriptor
    return sorted(
        prebuilt.values(),
        key=lambda descriptor: descriptor.key,
    )


def collect_runtime_topology_keys(
    vllm_config: Any,
) -> set[tuple[int, int, int, int, int, int]]:
    return {
        descriptor.key
        for descriptor in collect_runtime_topology_candidates(vllm_config)
    }


def _reject_unsupported_features(vllm_config: Any) -> None:
    parallel_config = vllm_config.parallel_config
    backend = parallel_config.distributed_executor_backend
    if backend not in ("mp", "uni", None):
        raise ValueError(
            "runtime topology switching currently supports only mp/uni "
            f"executors, got {backend!r}"
        )
    if parallel_config.data_parallel_size != 1:
        raise ValueError("runtime topology switching does not support data parallel")
    if (
        parallel_config.prefill_context_parallel_size != 1
        or (parallel_config.decode_context_parallel_size or 1) != 1
    ):
        raise ValueError(
            "runtime topology switching does not support context parallel"
        )
    if getattr(parallel_config, "enable_expert_parallel", False):
        raise ValueError(
            "runtime topology switching does not support expert parallel"
        )
    if getattr(parallel_config, "enable_elastic_ep", False):
        raise ValueError(
            "runtime topology switching does not support elastic expert parallel"
        )
    if vllm_config.lora_config is not None:
        raise ValueError("runtime topology switching does not support LoRA")
    if vllm_config.speculative_config is not None:
        raise ValueError(
            "runtime topology switching does not support speculative decoding"
        )
    if vllm_config.kv_transfer_config is not None:
        raise ValueError("runtime topology switching does not support KV transfer")
    if getattr(vllm_config, "ec_transfer_config", None) is not None:
        raise ValueError("runtime topology switching does not support EC transfer")
    if _getattr_nested(vllm_config, "cache_config.kv_offloading_size") is not None:
        raise ValueError("runtime topology switching does not support KV offload")
    if _getattr_nested(vllm_config, "offload_config.uva.cpu_offload_gb", 0) > 0:
        raise ValueError("runtime topology switching does not support weight offload")
    if _getattr_nested(vllm_config, "offload_config.prefetch.offload_group_size", 0) > 0:
        raise ValueError("runtime topology switching does not support weight offload")


def recommend_runtime_topology(
    vllm_config: Any,
    workload: RuntimeTopologyWorkload,
) -> RuntimeTopologyRecommendation:
    _reject_unsupported_features(vllm_config)

    previous = _current_topology(vllm_config)
    candidates = collect_runtime_topology_candidates(vllm_config)

    if workload.total_requests <= 1:
        reason = "low_concurrency"
        target = max(
            candidates,
            key=lambda descriptor: (
                descriptor.tensor_parallel_size,
                -descriptor.pipeline_parallel_size,
            ),
        )
    elif workload.total_requests >= previous.world_size:
        reason = "high_concurrency"
        target = max(
            candidates,
            key=lambda descriptor: (
                descriptor.pipeline_parallel_size,
                -descriptor.tensor_parallel_size,
            ),
        )
    else:
        reason = "balanced_concurrency"
        target = min(
            candidates,
            key=lambda descriptor: (
                abs(
                    descriptor.tensor_parallel_size
                    - descriptor.pipeline_parallel_size
                ),
                -descriptor.tensor_parallel_size,
            ),
        )

    return RuntimeTopologyRecommendation(
        previous_topology=previous,
        target_topology=target,
        workload=workload,
        reason=reason,
    )


def validate_runtime_topology_switch(
    vllm_config: Any,
    request: RuntimeTopologySwitchRequest,
    prebuilt_topology_keys: set[tuple[int, int, int, int, int, int]] | None = None,
) -> RuntimeTopologySwitchPlan:
    _reject_unsupported_features(vllm_config)

    previous = _current_topology(vllm_config)
    try:
        target = _target_topology(vllm_config, request)
    except ValueError as e:
        raise ValueError(
            "runtime topology switching currently requires the same world size"
        ) from e

    if target.world_size != previous.world_size:
        raise ValueError(
            "runtime topology switching currently requires the same world size"
        )
    prebuilt_keys = (
        collect_runtime_topology_keys(vllm_config)
        if prebuilt_topology_keys is None
        else prebuilt_topology_keys
    )
    if target.key not in prebuilt_keys:
        raise ValueError(
            "target topology must be prebuilt before runtime topology switching"
        )

    return RuntimeTopologySwitchPlan(
        previous_topology=previous,
        target_topology=target,
    )
