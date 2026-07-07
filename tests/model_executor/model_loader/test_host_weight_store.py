# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing as mp
from collections.abc import Iterable
from pathlib import Path

import pytest
import torch
from torch import nn

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.host_weight_store import HostWeightStore


def _example_weights() -> list[tuple[str, torch.Tensor]]:
    return [
        ("a.weight", torch.arange(12, dtype=torch.float32).reshape(3, 4)),
        ("b.weight", torch.arange(6, dtype=torch.float16).reshape(2, 3)),
        ("c.weight", torch.arange(5, dtype=torch.bfloat16)),
        ("d.weight", torch.arange(4, dtype=torch.int64)),
    ]


def _assert_weights_equal(
    actual: Iterable[tuple[str, torch.Tensor]],
    expected: Iterable[tuple[str, torch.Tensor]],
) -> None:
    actual_dict = dict(actual)
    expected_dict = dict(expected)
    assert list(actual_dict) == list(expected_dict)
    for name, expected_tensor in expected_dict.items():
        actual_tensor = actual_dict[name]
        assert actual_tensor.dtype == expected_tensor.dtype
        assert actual_tensor.shape == expected_tensor.shape
        assert torch.equal(actual_tensor, expected_tensor)


def test_host_weight_store_round_trips_checkpoint_tensors_from_mmap(tmp_path):
    store = HostWeightStore.build(
        _example_weights(),
        tmp_path / "weights",
    )

    reopened = HostWeightStore.open(store.metadata_path)

    _assert_weights_equal(reopened.iter_weights(), _example_weights())

    tensor = dict(reopened.iter_weights())["a.weight"]
    assert tensor.device.type == "cpu"
    assert tensor.untyped_storage().filename == str(store.data_path)


def test_host_weight_store_resolves_relative_data_path_from_metadata_dir(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    store = HostWeightStore.build(_example_weights(), "relative/weights")

    reopened = HostWeightStore.open("relative/weights.json")

    assert reopened.data_path == Path("relative/weights.bin")
    _assert_weights_equal(reopened.iter_weights(), _example_weights())


def test_host_weight_store_cleanup_removes_files(tmp_path):
    store = HostWeightStore.build(_example_weights(), tmp_path / "weights")
    data_path = store.data_path
    metadata_path = store.metadata_path

    store.close(remove_files=True)

    assert not data_path.exists()
    assert not metadata_path.exists()


def _open_store_worker(metadata_path: str, queue: mp.Queue) -> None:
    store = HostWeightStore.open(metadata_path)
    weights = list(store.iter_weights())
    queue.put(
        [
            (
                name,
                tensor.dtype,
                tuple(tensor.shape),
                tensor.untyped_storage().filename,
                float(tensor.float().sum()),
            )
            for name, tensor in weights
        ]
    )


def test_host_weight_store_can_be_opened_by_multiple_processes(tmp_path):
    store = HostWeightStore.build(_example_weights(), tmp_path / "weights")
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    processes = [
        ctx.Process(target=_open_store_worker, args=(str(store.metadata_path), queue))
        for _ in range(2)
    ]

    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join(timeout=30)

    assert all(proc.exitcode == 0 for proc in processes)
    results = [queue.get(timeout=5) for _ in processes]
    assert results[0] == results[1]
    assert all(entry[3] == str(store.data_path) for entry in results[0])


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loaded: list[tuple[str, torch.Tensor]] = []

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        self.loaded = [(name, tensor.clone()) for name, tensor in weights]
        return {name for name, _ in self.loaded}


def test_host_weight_store_loader_uses_store_iterator(tmp_path):
    store = HostWeightStore.build(_example_weights(), tmp_path / "weights")
    loader = get_model_loader(
        LoadConfig(
            load_format="host_weight_store",
            model_loader_extra_config={
                "metadata_path": str(store.metadata_path),
            },
        )
    )
    model = TinyModel()

    loader.load_weights(model, model_config=None)  # type: ignore[arg-type]

    _assert_weights_equal(model.loaded, _example_weights())


def test_host_weight_store_loader_requires_metadata_path():
    with pytest.raises(ValueError, match="metadata_path"):
        get_model_loader(LoadConfig(load_format="host_weight_store"))


def test_host_weight_store_loader_rejects_unsupported_extra_config(tmp_path):
    store = HostWeightStore.build(_example_weights(), tmp_path / "weights")
    with pytest.raises(ValueError, match="Unexpected extra config keys"):
        get_model_loader(
            LoadConfig(
                load_format="host_weight_store",
                model_loader_extra_config={
                    "metadata_path": str(store.metadata_path),
                    "unused": True,
                },
            )
        )
