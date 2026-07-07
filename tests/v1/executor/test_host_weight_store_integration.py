# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.config.load import LoadConfig
from vllm.v1.executor.host_weight_store_integration import (
    prepare_host_weight_store_load_config,
)


class FakeDefaultLoader:
    calls = 0

    def __init__(self, load_config):
        self.load_config = load_config

    def get_all_weights(self, model_config, model):
        type(self).calls += 1
        yield "a.weight", torch.arange(4, dtype=torch.float32)
        yield "b.weight", torch.arange(2, dtype=torch.float16)


def test_prepare_host_weight_store_load_config_builds_store_once(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "vllm.v1.executor.host_weight_store_integration.DefaultModelLoader",
        FakeDefaultLoader,
    )
    load_config = LoadConfig(
        load_format="safetensors",
        model_loader_extra_config={
            "host_weight_store_path": str(tmp_path / "shared-weights"),
        },
    )
    vllm_config = SimpleNamespace(
        load_config=load_config,
        model_config=SimpleNamespace(model="unused", revision=None),
    )

    prepared = prepare_host_weight_store_load_config(vllm_config)

    assert FakeDefaultLoader.calls == 1
    assert prepared.load_format == "host_weight_store"
    assert prepared.model_loader_extra_config == {
        "metadata_path": str(tmp_path / "shared-weights.json"),
    }
    assert (tmp_path / "shared-weights.bin").exists()
    assert (tmp_path / "shared-weights.json").exists()
    assert load_config.load_format == "safetensors"


def test_prepare_host_weight_store_load_config_noops_without_path():
    load_config = LoadConfig(load_format="safetensors")
    vllm_config = SimpleNamespace(load_config=load_config)

    assert prepare_host_weight_store_load_config(vllm_config) is load_config
