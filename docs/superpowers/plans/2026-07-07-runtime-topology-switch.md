# Runtime Topology Switch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement vLLM v1 runtime TP/PP topology switching for the same worker world size, discarding KV cache and recomputing unfinished requests.

**Architecture:** The switch is an EngineCore utility transaction: validate target topology, pause scheduling, reset unfinished requests for recompute, activate a prebuilt communication snapshot on workers, rebuild topology-dependent model/KV state, rebuild scheduler state, then resume. The first implementation supports single-node CUDA v1 `mp`, same `world_size`, no Ray/DP/LoRA/speculative decode/KV connector/KV offload/CUDA graph.

**Tech Stack:** Python, vLLM v1 EngineCore, MultiprocExecutor worker RPC, `TopologyDescriptor`, `HostWeightStore`, pytest, real 2-GPU NCCL smoke.

---

### Task 1: Runtime Topology Request Validation

**Files:**
- Create: `vllm/v1/engine/runtime_topology.py`
- Test: `tests/v1/engine/test_runtime_topology.py`

- [ ] **Step 1: Write failing validation tests**

Add tests that import `RuntimeTopologySwitchRequest` and `validate_runtime_topology_switch`, construct fake `VllmConfig` objects with `types.SimpleNamespace`, and assert:

```python
def test_validate_accepts_same_world_size_target_from_env(monkeypatch):
    monkeypatch.setenv("VLLM_PREBUILD_MODEL_PARALLEL_TOPOLOGIES", "tp=1,pp=2")
    config = make_config(tp=2, pp=1, world_size=2)
    result = validate_runtime_topology_switch(
        config, RuntimeTopologySwitchRequest(tensor_parallel_size=1,
                                             pipeline_parallel_size=2))
    assert result.previous_topology.tensor_parallel_size == 2
    assert result.target_topology.pipeline_parallel_size == 2
```

Also assert rejection for target not in env, world-size mismatch, Ray backend, DP > 1, DCP/PCP > 1, LoRA, speculative decode, KV transfer, EC transfer, KV offload, and CUDA graph mode not `NONE`.

- [ ] **Step 2: Run red test**

Run:

```bash
.venv312/bin/python -m pytest tests/v1/engine/test_runtime_topology.py -q
```

Expected: FAIL because `vllm.v1.engine.runtime_topology` does not exist.

- [ ] **Step 3: Implement validation module**

Create dataclasses:

```python
@dataclass(frozen=True)
class RuntimeTopologySwitchRequest:
    tensor_parallel_size: int
    pipeline_parallel_size: int

@dataclass(frozen=True)
class RuntimeTopologySwitchPlan:
    previous_topology: TopologyDescriptor
    target_topology: TopologyDescriptor
```

Implement `validate_runtime_topology_switch(vllm_config, request)` using `TopologyDescriptor` and `parse_topology_descriptors(envs.VLLM_PREBUILD_MODEL_PARALLEL_TOPOLOGIES, ...)`. Raise `ValueError` with explicit messages for unsupported features.

- [ ] **Step 4: Run green test**

Run:

```bash
.venv312/bin/python -m pytest tests/v1/engine/test_runtime_topology.py -q
```

Expected: PASS.

### Task 2: Worker and Executor Topology Rebuild Primitives

**Files:**
- Modify: `vllm/v1/executor/abstract.py`
- Modify: `vllm/v1/worker/gpu_worker.py`
- Modify: `vllm/v1/worker/gpu_model_runner.py`
- Test: `tests/v1/executor/test_runtime_topology_executor.py`

- [ ] **Step 1: Write failing executor tests**

Add fake executor and fake worker tests asserting:

```python
executor.activate_model_parallel_topology(descriptor)
executor.rebuild_model_for_runtime_topology()
executor.clear_runtime_kv_state()
```

delegate to worker RPC methods in order. Add GPUWorker tests with monkeypatched `parallel_state.activate_model_parallel_topology` and fake model runner to assert `clear_runtime_state_for_topology_rebuild(clear_model=True)` runs before `load_model()`.

- [ ] **Step 2: Run red test**

Run:

```bash
.venv312/bin/python -m pytest tests/v1/executor/test_runtime_topology_executor.py -q
```

Expected: FAIL because methods are missing.

- [ ] **Step 3: Implement minimal primitives**

Executor methods call collective RPC:

```python
def activate_model_parallel_topology(self, descriptor):
    self.collective_rpc("activate_model_parallel_topology", args=(descriptor,))

def rebuild_model_for_runtime_topology(self):
    self.collective_rpc("rebuild_model_for_runtime_topology")

def clear_runtime_kv_state(self):
    self.collective_rpc("clear_runtime_kv_state")
```

GPUWorker methods activate cached topology, clear model runner state, load model, and clear KV state.

- [ ] **Step 4: Run green test**

Run:

```bash
.venv312/bin/python -m pytest tests/v1/executor/test_runtime_topology_executor.py -q
```

Expected: PASS.

### Task 3: Scheduler Recompute Drain

**Files:**
- Modify: `vllm/v1/core/sched/scheduler.py`
- Test: `tests/v1/core/test_scheduler.py`

- [ ] **Step 1: Write failing scheduler test**

Add focused tests that create scheduler requests, call `drain_unfinished_requests_for_recompute(reset_running_requests=True)`, and assert exported requests have no KV/block-manager state and running requests are reset to recompute from token 0.

- [ ] **Step 2: Run red test**

Run:

```bash
.venv312/bin/python -m pytest tests/v1/core/test_scheduler.py -q -k runtime_topology
```

Expected: FAIL because the method is missing.

- [ ] **Step 3: Implement scheduler method**

Reuse `reset_prefix_cache(reset_running_requests=True)` for running request recompute semantics. Return unfinished `Request` objects in priority-preserving order, clear internal scheduler containers, and avoid migrating block ids or KV connector state.

- [ ] **Step 4: Run green test**

Run the same scheduler test. Expected: PASS.

### Task 4: EngineCore Switch Transaction and Public APIs

**Files:**
- Modify: `vllm/v1/engine/core.py`
- Modify: `vllm/v1/engine/core_client.py`
- Modify: `vllm/v1/engine/llm_engine.py`
- Modify: `vllm/v1/engine/async_llm.py`
- Modify: `vllm/entrypoints/llm.py`
- Test: `tests/v1/engine/test_runtime_topology.py`

- [ ] **Step 1: Write failing EngineCore API tests**

Add tests with fake scheduler/executor asserting transaction order:

```text
validate -> pause -> drain_recompute -> activate_topology -> rebuild_model
-> initialize_kv_caches -> rebuild_scheduler -> requeue_requests -> resume
```

Also assert `LLMEngine.switch_runtime_topology(tp, pp)` calls the EngineCore client utility.

- [ ] **Step 2: Run red test**

Run:

```bash
.venv312/bin/python -m pytest tests/v1/engine/test_runtime_topology.py -q
```

Expected: FAIL because switch methods are missing.

- [ ] **Step 3: Implement EngineCore switch**

Implement `EngineCore.switch_runtime_topology(request)` that validates, pauses, clears caches, activates worker topology, rebuilds model, reruns `_initialize_kv_caches`, reconstructs Scheduler with new KV config, re-adds drained requests with `num_computed_tokens=0`, and resumes. Add sync and async wrappers in clients and public engine classes.

- [ ] **Step 4: Run green test**

Run the same EngineCore test. Expected: PASS.

### Task 5: Distributed and Real GPU Verification

**Files:**
- Create: `tests/distributed/test_runtime_topology_switch.py`
- Create: `/home/mqy/vllm-tp-dp/stage3_runtime_topology_switch.md`

- [ ] **Step 1: Add distributed activation test**

Add a 2-process NCCL/Gloo test that prebuilds `TP=2,PP=1` and `TP=1,PP=2`, calls worker activation twice, and asserts no `new_group()` calls during activation.

- [ ] **Step 2: Add real 2-GPU smoke script to docs**

Document a command using local `gpt2`, `VLLM_PREBUILD_MODEL_PARALLEL_TOPOLOGIES=tp=1,pp=2`, `TP=2,PP=1 -> TP=1,PP=2 -> TP=2,PP=1`, one generation after each switch, and worker PID equality before/after switch.

- [ ] **Step 3: Run final verification**

Run:

```bash
.venv312/bin/python -m pytest \
  tests/v1/engine/test_runtime_topology.py \
  tests/v1/executor/test_runtime_topology_executor.py \
  tests/distributed/test_runtime_topology_switch.py \
  tests/distributed/test_topology_cache.py \
  tests/model_executor/model_loader/test_host_weight_store.py \
  tests/v1/executor/test_worker_lifecycle.py \
  -q
```

Then run the documented real GPU smoke under escalation. Expected: all tests pass and smoke proves same worker PIDs across switches.
