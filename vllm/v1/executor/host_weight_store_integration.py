# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Executor helpers for preparing file-backed host weight stores."""

from __future__ import annotations

import dataclasses
from typing import Any

from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.host_weight_store import HostWeightStore


def prepare_host_weight_store_load_config(vllm_config: Any):
    load_config = vllm_config.load_config
    extra_config = load_config.model_loader_extra_config
    if not isinstance(extra_config, dict):
        return load_config
    store_path = extra_config.get("host_weight_store_path")
    if not store_path:
        return load_config
    if not isinstance(store_path, str):
        raise ValueError("host_weight_store_path must be a non-empty string")

    source_extra_config = {
        key: value
        for key, value in extra_config.items()
        if key != "host_weight_store_path"
    }
    source_load_config = dataclasses.replace(
        load_config,
        model_loader_extra_config=source_extra_config,
    )
    source_loader = DefaultModelLoader(source_load_config)
    model = SimpleWeightIntrospectionModel()
    store = HostWeightStore.build(
        source_loader.get_all_weights(vllm_config.model_config, model),
        store_path,
    )
    return dataclasses.replace(
        load_config,
        load_format="host_weight_store",
        model_loader_extra_config={"metadata_path": str(store.metadata_path)},
    )


class SimpleWeightIntrospectionModel:
    """Minimal model shim for DefaultModelLoader.get_all_weights()."""

    fall_back_to_pt_during_load = True
    allow_patterns_overrides = None
    secondary_weights = ()
