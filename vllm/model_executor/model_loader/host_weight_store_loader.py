# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model loader that reads checkpoint tensors from a host weight store."""

from __future__ import annotations

from torch import nn

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.host_weight_store import HostWeightStore
from vllm.startup_profiling import startup_profile


class HostWeightStoreLoader(BaseModelLoader):
    """Load checkpoint-format tensors from a file-backed host store."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra_config = load_config.model_loader_extra_config
        if not isinstance(extra_config, dict):
            raise ValueError(
                "model_loader_extra_config must be a dict for "
                "load format host_weight_store"
            )
        allowed_keys = {"metadata_path"}
        unexpected_keys = set(extra_config) - allowed_keys
        if unexpected_keys:
            raise ValueError(
                "Unexpected extra config keys for load format "
                f"host_weight_store: {unexpected_keys}"
            )
        metadata_path = extra_config.get("metadata_path")
        if not isinstance(metadata_path, str) or not metadata_path:
            raise ValueError(
                "host_weight_store load format requires "
                "model_loader_extra_config['metadata_path']"
            )
        self.store = HostWeightStore.open(metadata_path)

    def download_model(self, model_config: ModelConfig) -> None:
        return

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        with startup_profile(
            "host_weight_store_iterator",
            tensor_count=len(self.store.tensor_metadata),
            metadata_path=str(self.store.metadata_path),
        ):
            model.load_weights(self.store.iter_weights())

    def get_all_weights(
        self,
        model_config: ModelConfig,
        model: nn.Module,
    ):
        yield from self.store.iter_weights()
