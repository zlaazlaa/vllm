# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import json
import os
from pathlib import Path

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput
from vllm.platforms import current_platform


DEFAULT_MODEL = (
    "/home/mqy/.cache/huggingface/hub/models--gpt2/"
    "snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"
)


def _runtime_topology_live_model() -> str:
    model = os.environ.get("VLLM_TEST_RUNTIME_TOPOLOGY_MODEL", DEFAULT_MODEL)
    if Path(model).exists():
        return model
    pytest.skip(
        "runtime topology live-KV smoke requires a local test model; set "
        "VLLM_TEST_RUNTIME_TOPOLOGY_MODEL"
    )


pytestmark = [
    pytest.mark.optional,
    pytest.mark.skipif(
        not current_platform.is_cuda(),
        reason="runtime topology live-KV smoke requires CUDA",
    ),
    pytest.mark.skipif(
        torch.accelerator.device_count() < 2,
        reason="runtime topology live-KV smoke requires 2 visible GPUs",
    ),
]


def _worker_topology(worker):
    from vllm.compilation.counter import compilation_counter
    from vllm.distributed import parallel_state

    return {
        "rank": worker.rank,
        "pid": os.getpid(),
        "tp": parallel_state.get_tp_group().world_size,
        "pp": parallel_state.get_pp_group().world_size,
        "captures": compilation_counter.num_gpu_runner_capture_triggers,
    }


def _assert_topology(llm: LLM, *, tp: int, pp: int) -> list[dict[str, int]]:
    workers = sorted(
        llm.collective_rpc(_worker_topology),
        key=lambda item: item["rank"],
    )
    assert len(workers) == 2
    for worker in workers:
        assert worker["tp"] == tp
        assert worker["pp"] == pp
    return workers


def _sampling_params() -> SamplingParams:
    return SamplingParams(
        temperature=0.0,
        max_tokens=8,
        ignore_eos=True,
    )


def _generated_token_ids(output: RequestOutput) -> tuple[int, ...]:
    assert len(output.outputs) == 1
    return tuple(output.outputs[0].token_ids)


def _step_until_partial_output(llm: LLM, request_id: str) -> RequestOutput:
    engine = llm.llm_engine
    for _ in range(4):
        outputs = engine.step()
        for output in outputs:
            if output.request_id == request_id:
                assert not output.finished
                assert _generated_token_ids(output)
                return output
        assert engine.has_unfinished_requests()
    raise AssertionError("request did not produce a partial output before switch")


def _drain_request(llm: LLM, request_id: str) -> RequestOutput:
    engine = llm.llm_engine
    final_output: RequestOutput | None = None
    for _ in range(32):
        if not engine.has_unfinished_requests():
            break
        for output in engine.step():
            if output.request_id == request_id:
                final_output = output
        if final_output is not None and final_output.finished:
            break
    assert final_output is not None
    assert final_output.finished
    return final_output


def _assert_capture_advanced(
    before: list[dict[str, int]],
    after: list[dict[str, int]],
) -> None:
    before_by_rank = {item["rank"]: item["captures"] for item in before}
    for item in after:
        assert item["captures"] > before_by_rank[item["rank"]]


def _make_llm(
    *,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    use_cuda_graph: bool,
    model_loader_extra_config: dict | None = None,
) -> LLM:
    kwargs = {}
    if use_cuda_graph:
        kwargs["compilation_config"] = {
            "mode": "NONE",
            "cudagraph_mode": "FULL",
            "cudagraph_capture_sizes": [1, 2],
        }
    return LLM(
        model=_runtime_topology_live_model(),
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        distributed_executor_backend="mp",
        enforce_eager=not use_cuda_graph,
        disable_custom_all_reduce=True,
        max_model_len=64,
        max_num_seqs=1,
        gpu_memory_utilization=0.30,
        dtype="float16",
        model_loader_extra_config=model_loader_extra_config or {},
        **kwargs,
    )


def _read_startup_profile_events(profile_dir: Path) -> list[dict]:
    events = []
    for profile_path in profile_dir.glob("startup_profile_*.jsonl"):
        with profile_path.open() as profile_file:
            for line in profile_file:
                if line.strip():
                    events.append(json.loads(line))
    return events


@pytest.mark.parametrize("use_cuda_graph", [False, True], ids=["eager", "cudagraph"])
@pytest.mark.parametrize(
    ("initial_tp", "initial_pp", "target_tp", "target_pp"),
    [
        pytest.param(2, 1, 1, 2, id="tp2pp1-to-tp1pp2"),
        pytest.param(1, 2, 2, 1, id="tp1pp2-to-tp2pp1"),
    ],
)
def test_runtime_topology_switch_migrates_live_kv_request(
    monkeypatch,
    use_cuda_graph: bool,
    initial_tp: int,
    initial_pp: int,
    target_tp: int,
    target_pp: int,
):
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    monkeypatch.setenv(
        "VLLM_PREBUILD_MODEL_PARALLEL_TOPOLOGIES",
        f"tp={target_tp},pp={target_pp}",
    )
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")

    prompt = "The capital of France is"
    llm = _make_llm(
        tensor_parallel_size=initial_tp,
        pipeline_parallel_size=initial_pp,
        use_cuda_graph=use_cuda_graph,
    )
    try:
        before = _assert_topology(llm, tp=initial_tp, pp=initial_pp)
        baseline = llm.generate([prompt], _sampling_params())[0]
        baseline_token_ids = _generated_token_ids(baseline)

        request_id = "runtime-topology-live-kv"
        llm.llm_engine.add_request(request_id, prompt, _sampling_params())
        partial_output = _step_until_partial_output(llm, request_id)

        result = llm.switch_runtime_topology(
            tensor_parallel_size=target_tp,
            pipeline_parallel_size=target_pp,
            max_kv_migration_blocks_per_step=2,
        )

        middle = _assert_topology(llm, tp=target_tp, pp=target_pp)
        assert [item["pid"] for item in middle] == [
            item["pid"] for item in before
        ]
        if use_cuda_graph:
            _assert_capture_advanced(before, middle)
        assert result["kv_cache_migration"]["policy"] == "migrate"
        assert result["kv_cache_migration"]["reason"] == "capacity_available"
        assert result["kv_cache_migration"]["request_state"] == "migrated"
        assert result["kv_cache_migration"]["max_blocks_per_step"] == 2
        assert result["kv_cache_migration"]["live_blocks"] > 0

        final_output = _drain_request(llm, request_id)
        assert _generated_token_ids(partial_output)
        assert _generated_token_ids(final_output) == baseline_token_ids
    finally:
        del llm
        gc.collect()


@pytest.mark.parametrize("use_cuda_graph", [False, True], ids=["eager", "cudagraph"])
def test_runtime_topology_switch_runs_consecutive_bidirectional_switches(
    monkeypatch,
    use_cuda_graph: bool,
):
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    monkeypatch.setenv(
        "VLLM_PREBUILD_MODEL_PARALLEL_TOPOLOGIES",
        "tp=1,pp=2;tp=2,pp=1",
    )
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")

    prompt = "The capital of France is"
    llm = _make_llm(
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        use_cuda_graph=use_cuda_graph,
    )
    try:
        before = _assert_topology(llm, tp=2, pp=1)
        first = llm.generate([prompt], _sampling_params())[0]
        first_token_ids = _generated_token_ids(first)

        switch_to_pp = llm.switch_runtime_topology(
            tensor_parallel_size=1,
            pipeline_parallel_size=2,
        )
        middle = _assert_topology(llm, tp=1, pp=2)
        assert [item["pid"] for item in middle] == [
            item["pid"] for item in before
        ]
        if use_cuda_graph:
            _assert_capture_advanced(before, middle)
        assert switch_to_pp["kv_cache_migration"]["policy"] == "recompute"
        assert switch_to_pp["kv_cache_migration"]["reason"] == "no_live_blocks"

        second = llm.generate([prompt], _sampling_params())[0]
        assert _generated_token_ids(second) == first_token_ids

        switch_to_tp = llm.switch_runtime_topology(
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
        )
        after = _assert_topology(llm, tp=2, pp=1)
        assert [item["pid"] for item in after] == [
            item["pid"] for item in before
        ]
        if use_cuda_graph:
            _assert_capture_advanced(middle, after)
        assert switch_to_tp["kv_cache_migration"]["policy"] == "recompute"
        assert switch_to_tp["kv_cache_migration"]["reason"] == "no_live_blocks"

        third = llm.generate([prompt], _sampling_params())[0]
        assert _generated_token_ids(third) == first_token_ids
    finally:
        del llm
        gc.collect()


def test_runtime_topology_switch_uses_host_weight_store_for_rebuild(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    monkeypatch.setenv(
        "VLLM_PREBUILD_MODEL_PARALLEL_TOPOLOGIES",
        "tp=1,pp=2",
    )
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")
    profile_dir = tmp_path / "profiles"
    monkeypatch.setenv("VLLM_STARTUP_PROFILING", "1")
    monkeypatch.setenv("VLLM_STARTUP_PROFILE_DIR", str(profile_dir))

    prompt = "The capital of France is"
    llm = _make_llm(
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        use_cuda_graph=False,
        model_loader_extra_config={
            "host_weight_store_path": str(tmp_path / "shared-weights"),
        },
    )
    try:
        baseline = llm.generate([prompt], _sampling_params())[0]
        baseline_token_ids = _generated_token_ids(baseline)
        host_store_loads_before_switch = [
            event
            for event in _read_startup_profile_events(profile_dir)
            if event.get("name") == "host_weight_store_iterator"
        ]

        result = llm.switch_runtime_topology(
            tensor_parallel_size=1,
            pipeline_parallel_size=2,
        )

        assert result["model_materialization"] == {
            "load_format": "host_weight_store",
            "uses_host_weight_store": True,
        }
        _assert_topology(llm, tp=1, pp=2)
        after = llm.generate([prompt], _sampling_params())[0]
        assert _generated_token_ids(after) == baseline_token_ids
    finally:
        del llm
        gc.collect()

    events = _read_startup_profile_events(profile_dir)
    host_store_loads = [
        event
        for event in events
        if event.get("name") == "host_weight_store_iterator"
    ]
    assert len(host_store_loads) - len(host_store_loads_before_switch) >= 2
