# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import replace
from functools import cached_property
from typing import TYPE_CHECKING, Literal, TypeVar, overload

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.topology_cache import TopologyDescriptor
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.tasks import SupportedTask
from vllm.tracing import instrument
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.core.kv_cache_migration import (
    RuntimeKVHeadPartition,
    RuntimeKVLayerPartition,
    RuntimeKVMigrationPlan,
)
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.engine import ReconfigureDistributedRequest
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase

logger = init_logger(__name__)

_R = TypeVar("_R")

FailureCallback = Callable[[], None]


class Executor(ABC):
    """Abstract base class for vLLM executors."

    An executor is responsible for executing the model on one device,
    or it can be a distributed executor that can execute the model on multiple devices.
    """

    uses_ray: bool = False  # whether the executor uses Ray for orchestration.
    supports_pp: bool = False  # whether the executor supports PP

    @staticmethod
    def get_class(vllm_config: VllmConfig) -> type["Executor"]:
        executor_class: type[Executor]
        parallel_config = vllm_config.parallel_config
        distributed_executor_backend = parallel_config.distributed_executor_backend
        # distributed_executor_backend must be set in VllmConfig.__post_init__
        if isinstance(distributed_executor_backend, type):
            if not issubclass(distributed_executor_backend, Executor):
                raise TypeError(
                    "distributed_executor_backend must be a subclass of "
                    f"Executor. Got {distributed_executor_backend}."
                )
            executor_class = distributed_executor_backend
        elif distributed_executor_backend == "ray":
            if envs.VLLM_USE_RAY_V2_EXECUTOR_BACKEND:
                from vllm.v1.executor.ray_executor_v2 import RayExecutorV2

                executor_class = RayExecutorV2
            else:
                from vllm.v1.executor.ray_executor import RayDistributedExecutor

                executor_class = RayDistributedExecutor
        elif distributed_executor_backend == "mp":
            from vllm.v1.executor.multiproc_executor import MultiprocExecutor

            executor_class = MultiprocExecutor
        elif distributed_executor_backend == "uni":
            from vllm.v1.executor.uniproc_executor import UniProcExecutor

            executor_class = UniProcExecutor
        elif distributed_executor_backend == "external_launcher":
            # TODO: make v1 scheduling deterministic
            # to support external launcher
            executor_class = ExecutorWithExternalLauncher
        elif isinstance(distributed_executor_backend, str):
            executor_class = resolve_obj_by_qualname(distributed_executor_backend)
            if not issubclass(executor_class, Executor):
                raise TypeError(
                    "distributed_executor_backend must be a subclass of "
                    f"Executor. Got {executor_class}."
                )
        else:
            raise ValueError(
                f"Unknown distributed executor backend: {distributed_executor_backend}"
            )
        return executor_class

    @instrument(span_name="Executor init")
    def __init__(
        self,
        vllm_config: VllmConfig,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config
        self._init_executor()
        self.is_sleeping = False
        self.sleeping_tags: set[str] = set()
        self.kv_output_aggregator: KVOutputAggregator | None = None

    @abstractmethod
    def _init_executor(self) -> None:
        raise NotImplementedError

    def initialize_from_config(self, kv_cache_configs: list[KVCacheConfig]) -> None:
        """
        Initialize the KV caches and begin the model execution loop of the
        underlying workers.
        """
        self.collective_rpc("initialize_from_config", args=(kv_cache_configs,))
        compilation_times: list[CompilationTimes] = self.collective_rpc(
            "compile_or_warm_up_model"
        )
        # Propagate compilation time from workers back to the main process.
        # With TP>1, compilation happens in worker processes, so the main
        # process config is never updated. Use max across workers since they
        # compile in parallel.
        if compilation_times:
            self.vllm_config.compilation_config.compilation_time = max(
                t.language_model for t in compilation_times
            )
            self.vllm_config.compilation_config.encoder_compilation_time = max(
                t.encoder for t in compilation_times
            )

    def register_failure_callback(self, callback: FailureCallback):  # noqa: B027
        """
        Register a function to be called if the executor enters a permanent
        failed state.
        """
        pass

    def determine_available_memory(self) -> list[int]:  # in bytes
        return self.collective_rpc("determine_available_memory")

    def get_kv_cache_specs(self) -> list[dict[str, KVCacheSpec]]:
        return self.collective_rpc("get_kv_cache_spec")

    @overload
    def collective_rpc(
        self,
        method: str | Callable[[WorkerBase], _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: Literal[False] = False,
    ) -> list[_R]:
        """
        Execute an RPC call on all workers.

        Args:
            method: Name of the worker method to execute, or a callable that
                is serialized and sent to all workers to execute.

                If the method is a callable, it should accept an additional
                `self` argument, in addition to the arguments passed in `args`
                and `kwargs`. The `self` argument will be the worker object.
            timeout: Maximum time in seconds to wait for execution. Raises a
                [`TimeoutError`][] on timeout. `None` means wait indefinitely.
            args: Positional arguments to pass to the worker method.
            kwargs: Keyword arguments to pass to the worker method.
            non_block: If `True`, returns a list of Futures instead of waiting
                for the results.

        Returns:
            A list containing the results from each worker.

        Note:
            It is recommended to use this API to only pass control messages,
            and set up data-plane communication to pass data.
        """
        pass

    @overload
    def collective_rpc(
        self,
        method: str | Callable[[WorkerBase], _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: Literal[True] = True,
    ) -> Future[list[_R]]:
        pass

    @abstractmethod
    def collective_rpc(
        self, method, timeout=None, args=(), kwargs=None, non_block: bool = False
    ):
        raise NotImplementedError

    def get_kv_connector_handshake_metadata(
        self,
    ) -> list[dict[tuple[int, int], KVConnectorHandshakeMetadata]]:
        return self.collective_rpc("get_kv_connector_handshake_metadata")

    @overload
    def execute_model(
        self, scheduler_output: SchedulerOutput, non_block: Literal[False] = False
    ) -> ModelRunnerOutput | None:
        pass

    @overload
    def execute_model(
        self, scheduler_output: SchedulerOutput, non_block: Literal[True] = True
    ) -> Future[ModelRunnerOutput | None]:
        pass

    def execute_model(
        self, scheduler_output: SchedulerOutput, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        output = self.collective_rpc(  # type: ignore[call-overload]
            "execute_model", args=(scheduler_output,), non_block=non_block
        )
        return output[0]

    @overload
    def sample_tokens(
        self, grammar_output: GrammarOutput | None, non_block: Literal[False] = False
    ) -> ModelRunnerOutput:
        pass

    @overload
    def sample_tokens(
        self, grammar_output: GrammarOutput | None, non_block: Literal[True] = True
    ) -> Future[ModelRunnerOutput]:
        pass

    def sample_tokens(
        self, grammar_output: GrammarOutput | None, non_block: bool = False
    ) -> ModelRunnerOutput | Future[ModelRunnerOutput]:
        output = self.collective_rpc(  # type: ignore[call-overload]
            "sample_tokens", args=(grammar_output,), non_block=non_block
        )
        return output[0]

    def execute_dummy_batch(self) -> None:
        self.collective_rpc("execute_dummy_batch")

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        output: list[DraftTokenIds] = self.collective_rpc("take_draft_token_ids")
        return output[0]

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        self.collective_rpc("profile", args=(is_start, profile_prefix))

    def save_sharded_state(
        self,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        self.collective_rpc(
            "save_sharded_state",
            kwargs=dict(path=path, pattern=pattern, max_size=max_size),
        )

    def activate_model_parallel_topology(
        self,
        descriptor: TopologyDescriptor,
    ) -> None:
        self.collective_rpc(
            "activate_model_parallel_topology",
            args=(descriptor,),
        )

    def update_runtime_topology_config(
        self,
        descriptor: TopologyDescriptor,
    ) -> None:
        self.collective_rpc(
            "update_runtime_topology_config",
            args=(descriptor,),
        )

    def rebuild_model_for_runtime_topology(self) -> None:
        self.collective_rpc("rebuild_model_for_runtime_topology")

    def snapshot_runtime_kv_caches_for_topology_migration(self) -> None:
        self.collective_rpc("snapshot_runtime_kv_caches_for_topology_migration")

    def clear_runtime_kv_migration_snapshot(self) -> None:
        self.collective_rpc("clear_runtime_kv_migration_snapshot")

    def clear_runtime_kv_state(self) -> None:
        self.collective_rpc("clear_runtime_kv_state")

    def migrate_runtime_kv_cache_for_topology(
        self,
        *,
        plan: RuntimeKVMigrationPlan,
        block_mapping: dict[int, int],
        max_blocks_per_step: int = 1,
    ) -> dict[str, int]:
        if max_blocks_per_step < 1:
            raise ValueError("max_blocks_per_step must be >= 1")

        migration_steps = 0
        tensor_copies = 0
        source_shard_keys: set[tuple[int, int]] = set()
        source_block_ids = tuple(block_mapping)

        def collect_source_kv_caches(
            *,
            layer_names: tuple[str, ...],
            block_ids: tuple[int, ...],
            head_indices: tuple[int, ...],
        ) -> dict[tuple[int, int], object]:
            kwargs: dict[str, object] = {"layer_names": layer_names}
            if block_ids:
                kwargs["block_ids"] = block_ids
            if head_indices:
                kwargs["head_indices"] = head_indices
            source_shards = self.collective_rpc(
                "export_runtime_kv_source_shard_for_migration",
                args=(plan,),
                kwargs=kwargs,
            )
            source_kv_caches = {}
            for shard in source_shards:
                if shard is None:
                    continue
                shard_key = (shard["pp_rank"], shard["tp_rank"])
                if shard_key in source_kv_caches:
                    raise ValueError(
                        "duplicate runtime KV source shard for "
                        f"pp={shard_key[0]}, tp={shard_key[1]}"
                    )
                source_kv_caches[shard_key] = shard["kv_caches"]
            if not source_kv_caches:
                raise ValueError("runtime KV migration has no source shards")
            return source_kv_caches

        for pp_partition in plan.pp_partitions:
            for layer_index in pp_partition.layer_indices:
                layer_names = (plan.layer_names[layer_index],)
                layer_partition = RuntimeKVLayerPartition(
                    pp_rank=pp_partition.pp_rank,
                    layer_indices=range(layer_index, layer_index + 1),
                )
                block_batches = [
                    source_block_ids[
                        block_start : block_start + max_blocks_per_step
                    ]
                    for block_start in range(
                        0,
                        len(source_block_ids),
                        max_blocks_per_step,
                    )
                ] or [()]
                for block_ids in block_batches:
                    for tp_partition in plan.tp_partitions:
                        source_kv_caches = collect_source_kv_caches(
                            layer_names=layer_names,
                            block_ids=block_ids,
                            head_indices=tuple(tp_partition.head_indices),
                        )

                        source_shard_keys.update(source_kv_caches)
                        batch_block_mapping = {
                            local_block_id: block_mapping[source_block_id]
                            for local_block_id, source_block_id in enumerate(
                                block_ids
                            )
                        }
                        head_partition = RuntimeKVHeadPartition(
                            tp_rank=tp_partition.tp_rank,
                            head_indices=range(
                                tp_partition.head_indices.start,
                                tp_partition.head_indices.stop,
                            ),
                        )
                        batch_plan = (
                            plan
                            if (
                                len(block_ids) == plan.live_blocks
                                and len(source_block_ids) == plan.live_blocks
                                and source_block_ids == block_ids
                                and batch_block_mapping == block_mapping
                                and len(plan.pp_partitions) == 1
                                and pp_partition.layer_indices
                                == layer_partition.layer_indices
                                and len(plan.tp_partitions) == 1
                                and tp_partition.head_indices
                                == head_partition.head_indices
                            )
                            else replace(
                                plan,
                                pp_partitions=[layer_partition],
                                tp_partitions=[head_partition],
                                live_blocks=len(block_ids),
                            )
                        )
                        worker_stats = self.collective_rpc(
                            "migrate_runtime_kv_cache_for_topology",
                            kwargs=dict(
                                plan=batch_plan,
                                source_kv_caches=source_kv_caches,
                                block_mapping=batch_block_mapping,
                                max_blocks_per_step=max_blocks_per_step,
                            ),
                        )
                        migration_steps += sum(
                            int(stats["migration_steps"])
                            for stats in worker_stats
                        )
                        tensor_copies += sum(
                            int(stats["tensor_copies"]) for stats in worker_stats
                        )

        if not source_shard_keys:
            raise ValueError("runtime KV migration has no source shards")

        return {
            "migration_steps": migration_steps,
            "tensor_copies": tensor_copies,
            "source_shards": len(source_shard_keys),
        }

    def migrate_runtime_kv_cache_for_topology_p2p(
        self,
        *,
        plan: RuntimeKVMigrationPlan,
        block_mapping: dict[int, int],
        max_blocks_per_step: int = 1,
    ) -> dict[str, int]:
        if max_blocks_per_step < 1:
            raise ValueError("max_blocks_per_step must be >= 1")

        migration_steps = 0
        tensor_copies = 0
        p2p_sends = 0
        p2p_recvs = 0
        source_block_ids = tuple(block_mapping)

        for pp_partition in plan.pp_partitions:
            for layer_index in pp_partition.layer_indices:
                layer_partition = RuntimeKVLayerPartition(
                    pp_rank=pp_partition.pp_rank,
                    layer_indices=range(layer_index, layer_index + 1),
                )
                block_batches = [
                    source_block_ids[
                        block_start : block_start + max_blocks_per_step
                    ]
                    for block_start in range(
                        0,
                        len(source_block_ids),
                        max_blocks_per_step,
                    )
                ] or [()]
                for block_ids in block_batches:
                    batch_block_mapping = {
                        local_block_id: block_mapping[source_block_id]
                        for local_block_id, source_block_id in enumerate(
                            block_ids
                        )
                    }
                    for tp_partition in plan.tp_partitions:
                        head_partition = RuntimeKVHeadPartition(
                            tp_rank=tp_partition.tp_rank,
                            head_indices=range(
                                tp_partition.head_indices.start,
                                tp_partition.head_indices.stop,
                            ),
                        )
                        batch_plan = replace(
                            plan,
                            pp_partitions=[layer_partition],
                            tp_partitions=[head_partition],
                            live_blocks=len(block_ids),
                        )
                        worker_stats = self.collective_rpc(
                            "migrate_runtime_kv_cache_for_topology_p2p",
                            kwargs=dict(
                                plan=batch_plan,
                                block_mapping=batch_block_mapping,
                                source_block_ids=block_ids,
                                max_blocks_per_step=max_blocks_per_step,
                            ),
                        )
                        migration_steps += sum(
                            int(stats["migration_steps"])
                            for stats in worker_stats
                        )
                        tensor_copies += sum(
                            int(stats["tensor_copies"]) for stats in worker_stats
                        )
                        p2p_sends += sum(
                            int(stats["p2p_sends"]) for stats in worker_stats
                        )
                        p2p_recvs += sum(
                            int(stats["p2p_recvs"]) for stats in worker_stats
                        )

        return {
            "migration_steps": migration_steps,
            "tensor_copies": tensor_copies,
            "p2p_sends": p2p_sends,
            "p2p_recvs": p2p_recvs,
        }

    @abstractmethod
    def check_health(self) -> None:
        """Checks if the executor is healthy. If not, it should raise an
        exception."""
        raise NotImplementedError

    def shutdown(self) -> None:
        """Shutdown the executor."""
        self.collective_rpc("shutdown")

    def init_kv_output_aggregator(self, connector: "KVConnectorBase") -> None:
        """Init KVOutputAggregator"""
        self.kv_output_aggregator = KVOutputAggregator.from_connector(
            connector, self.parallel_config.world_size
        )

    @cached_property  # Avoid unnecessary RPC calls
    def supported_tasks(self) -> tuple[SupportedTask, ...]:
        output: list[tuple[SupportedTask, ...]]
        output = self.collective_rpc("get_supported_tasks")
        return output[0]

    def add_lora(self, lora_request: LoRARequest) -> bool:
        assert lora_request.lora_int_id > 0, "lora_id must be greater than 0."
        return all(self.collective_rpc("add_lora", args=(lora_request,)))

    def remove_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        return all(self.collective_rpc("remove_lora", args=(lora_id,)))

    def pin_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        return all(self.collective_rpc("pin_lora", args=(lora_id,)))

    def list_loras(self) -> set[int]:
        sets: list[set[int]] = self.collective_rpc("list_loras")
        for s in sets:
            assert s == sets[0], "All workers should have the same LORAs."
        return sets[0]

    def reset_mm_cache(self) -> None:
        """Reset the multi-modal cache in each worker."""
        self.collective_rpc("reset_mm_cache")

    def reset_encoder_cache(self) -> None:
        """Reset the encoder cache in each worker to clear cached encoder outputs."""
        self.collective_rpc("reset_encoder_cache")

    def sleep(self, level: int = 1):
        if self.is_sleeping:
            logger.warning("Executor is already sleeping.")
            return
        time_before_sleep = time.perf_counter()
        self.collective_rpc("sleep", kwargs=dict(level=level))
        time_after_sleep = time.perf_counter()
        self.sleeping_tags = {"weights", "kv_cache"}
        self.is_sleeping = True
        logger.info(
            "It took %.6f seconds to fall asleep.", time_after_sleep - time_before_sleep
        )

    def wake_up(self, tags: list[str] | None = None):
        if not self.is_sleeping:
            logger.warning("Executor is not sleeping.")
            return
        if tags:
            for tag in tags:
                if tag not in self.sleeping_tags:
                    logger.warning(
                        "Tag %s is not in sleeping tags %s", tag, self.sleeping_tags
                    )
                    return
        time_before_wakeup = time.perf_counter()
        self.collective_rpc("wake_up", kwargs=dict(tags=tags))
        time_after_wakeup = time.perf_counter()
        logger.info(
            "It took %.6f seconds to wake up tags %s.",
            time_after_wakeup - time_before_wakeup,
            tags if tags is not None else self.sleeping_tags,
        )
        if tags:
            for tag in tags:
                self.sleeping_tags.remove(tag)
        else:
            self.sleeping_tags.clear()
        if not self.sleeping_tags:
            self.is_sleeping = False

    def reinitialize_distributed(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        raise NotImplementedError

    @classmethod
    def supports_async_scheduling(cls) -> bool:
        """
        Whether the executor supports async scheduling.
        """
        return False


from vllm.v1.executor.uniproc_executor import (  # noqa: E402
    ExecutorWithExternalLauncher as _ExecutorWithExternalLauncher,
)
from vllm.v1.executor.uniproc_executor import (  # noqa: E402
    UniProcExecutor as _UniProcExecutor,
)

# For backwards compatibility.
UniProcExecutor = _UniProcExecutor
ExecutorWithExternalLauncher = _ExecutorWithExternalLauncher
