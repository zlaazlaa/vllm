# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""File-backed host memory store for checkpoint-format model weights."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


_DTYPE_TO_NAME: dict[torch.dtype, str] = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uint8",
    torch.bool: "bool",
}
_NAME_TO_DTYPE = {name: dtype for dtype, name in _DTYPE_TO_NAME.items()}


def _dtype_name(dtype: torch.dtype) -> str:
    try:
        return _DTYPE_TO_NAME[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported host weight dtype: {dtype}") from exc


def _dtype_from_name(name: str) -> torch.dtype:
    try:
        return _NAME_TO_DTYPE[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported host weight dtype in metadata: {name}") from exc


@dataclass(frozen=True)
class HostWeightTensorMetadata:
    name: str
    dtype: torch.dtype
    shape: tuple[int, ...]
    offset_bytes: int
    nbytes: int

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": _dtype_name(self.dtype),
            "shape": list(self.shape),
            "offset_bytes": self.offset_bytes,
            "nbytes": self.nbytes,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "HostWeightTensorMetadata":
        return cls(
            name=str(data["name"]),
            dtype=_dtype_from_name(str(data["dtype"])),
            shape=tuple(int(dim) for dim in data["shape"]),
            offset_bytes=int(data["offset_bytes"]),
            nbytes=int(data["nbytes"]),
        )


class HostWeightStore:
    """Checkpoint-format CPU tensor store backed by a single mmap file."""

    METADATA_VERSION = 1

    def __init__(
        self,
        metadata_path: str | os.PathLike[str],
        data_path: str | os.PathLike[str],
        tensors: Iterable[HostWeightTensorMetadata],
    ) -> None:
        self.metadata_path = Path(metadata_path)
        self.data_path = Path(data_path)
        self._tensors = tuple(tensors)
        self._storage: torch.UntypedStorage | None = None

    @property
    def tensor_metadata(self) -> tuple[HostWeightTensorMetadata, ...]:
        return self._tensors

    @classmethod
    def build(
        cls,
        weights: Iterable[tuple[str, torch.Tensor]],
        store_path: str | os.PathLike[str],
    ) -> "HostWeightStore":
        base_path = Path(store_path)
        base_path.parent.mkdir(parents=True, exist_ok=True)
        data_path = base_path.with_suffix(".bin")
        metadata_path = base_path.with_suffix(".json")
        tmp_data_path = data_path.with_name(f"{data_path.name}.{uuid.uuid4()}.tmp")
        tmp_metadata_path = metadata_path.with_name(
            f"{metadata_path.name}.{uuid.uuid4()}.tmp"
        )

        tensors: list[HostWeightTensorMetadata] = []
        offset = 0
        try:
            with tmp_data_path.open("wb") as data_file:
                for name, tensor in weights:
                    cpu_tensor = tensor.detach().cpu().contiguous()
                    element_size = cpu_tensor.element_size()
                    aligned_offset = _align_offset(offset, element_size)
                    if aligned_offset > offset:
                        data_file.write(b"\0" * (aligned_offset - offset))
                        offset = aligned_offset
                    nbytes = cpu_tensor.numel() * cpu_tensor.element_size()
                    data_file.write(cpu_tensor.view(torch.uint8).numpy().tobytes())
                    tensors.append(
                        HostWeightTensorMetadata(
                            name=name,
                            dtype=cpu_tensor.dtype,
                            shape=tuple(cpu_tensor.shape),
                            offset_bytes=offset,
                            nbytes=nbytes,
                        )
                    )
                    offset += nbytes

            metadata = {
                "version": cls.METADATA_VERSION,
                "data_path": os.path.relpath(data_path, metadata_path.parent),
                "tensors": [tensor.to_json() for tensor in tensors],
            }
            tmp_metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            os.replace(tmp_data_path, data_path)
            os.replace(tmp_metadata_path, metadata_path)
        finally:
            tmp_data_path.unlink(missing_ok=True)
            tmp_metadata_path.unlink(missing_ok=True)

        logger.info(
            "Built host weight store at %s with %d tensors and %.2f MiB",
            metadata_path,
            len(tensors),
            offset / (1024 * 1024),
        )
        return cls(metadata_path, data_path, tensors)

    @classmethod
    def open(
        cls,
        metadata_path: str | os.PathLike[str],
    ) -> "HostWeightStore":
        metadata_path = Path(metadata_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        version = int(metadata.get("version", 0))
        if version != cls.METADATA_VERSION:
            raise ValueError(
                f"Unsupported host weight store metadata version: {version}"
            )
        data_path = Path(metadata["data_path"])
        if not data_path.is_absolute():
            data_path = metadata_path.parent / data_path
        tensors = [
            HostWeightTensorMetadata.from_json(tensor)
            for tensor in metadata["tensors"]
        ]
        return cls(metadata_path, data_path, tensors)

    def _get_storage(self) -> torch.UntypedStorage:
        if self._storage is None:
            nbytes = self.data_path.stat().st_size
            self._storage = torch.UntypedStorage.from_file(
                str(self.data_path),
                shared=True,
                nbytes=nbytes,
            )
        return self._storage

    def iter_weights(self) -> Iterator[tuple[str, torch.Tensor]]:
        storage = self._get_storage()
        for metadata in self._tensors:
            tensor = torch.empty(0, dtype=metadata.dtype)
            element_size = tensor.element_size()
            if metadata.offset_bytes % element_size != 0:
                raise ValueError(
                    f"Tensor {metadata.name} has unaligned offset "
                    f"{metadata.offset_bytes} for dtype {metadata.dtype}"
                )
            tensor.set_(
                storage,
                metadata.offset_bytes // element_size,
                metadata.shape,
                _contiguous_stride(metadata.shape),
            )
            yield metadata.name, tensor

    def close(self, *, remove_files: bool = False) -> None:
        self._storage = None
        if remove_files:
            self.data_path.unlink(missing_ok=True)
            self.metadata_path.unlink(missing_ok=True)


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride: list[int] = []
    running = 1
    for size in reversed(shape):
        stride.append(running)
        running *= size
    return tuple(reversed(stride))


def _align_offset(offset: int, alignment: int) -> int:
    remainder = offset % alignment
    if remainder == 0:
        return offset
    return offset + alignment - remainder
