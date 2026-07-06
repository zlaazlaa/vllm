# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import vllm.envs as envs

_events: list[dict[str, Any]] = []
_lock = threading.Lock()


def get_startup_profile_events() -> list[dict[str, Any]]:
    with _lock:
        return list(_events)


def reset_startup_profile_events() -> None:
    with _lock:
        _events.clear()


def _profile_path() -> str | None:
    profile_dir = envs.VLLM_STARTUP_PROFILE_DIR
    if not profile_dir:
        return None
    os.makedirs(profile_dir, exist_ok=True)
    return os.path.join(profile_dir, f"startup_profile_{os.getpid()}.jsonl")


def _write_event(event: dict[str, Any]) -> None:
    path = _profile_path()
    if path is None:
        return
    with open(path, "a") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


@contextmanager
def startup_profile(name: str, **metadata: Any) -> Iterator[None]:
    if not envs.VLLM_STARTUP_PROFILING:
        yield
        return

    start_s = time.perf_counter()
    try:
        yield
    finally:
        end_s = time.perf_counter()
        event = {
            "name": name,
            "pid": os.getpid(),
            "thread_id": threading.get_ident(),
            "start_s": start_s,
            "end_s": end_s,
            "duration_s": end_s - start_s,
            "metadata": metadata,
        }
        with _lock:
            _events.append(event)
        _write_event(event)
