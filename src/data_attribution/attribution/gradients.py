"""Load query gradients from files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from data_attribution.attribution.types import QueryGradient


def _load_gradient_file(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Gradient file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".pt":
        loaded = torch.load(path)
        if not torch.is_tensor(loaded):
            raise TypeError(f"Expected torch.Tensor from {path}, got {type(loaded)}")
        return loaded
    if suffix == ".npy":
        array = np.load(path)
        return torch.from_numpy(array)
    raise ValueError(f"Unsupported gradient file format: {path}")


def _coerce_gradient(
    value: object,
    *,
    gradient_path: Path | None = None,
) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if gradient_path is not None:
        return _load_gradient_file(gradient_path)
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, (list, tuple)):
        return torch.tensor(value, dtype=torch.float32)
    raise TypeError("Gradient value must be a tensor, list, tuple, or ndarray")


def _load_jsonl_gradients(path: Path) -> list[QueryGradient]:
    records: list[QueryGradient] = []
    with path.open("r", encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "query_id" not in record or record["query_id"] is None:
                raise ValueError(
                    f"Missing query_id in gradient record at {path}:{lineno}"
                )
            query_id = str(record["query_id"])
            gradient_path = record.get("gradient_path")
            metadata = record.get("metadata")
            gradient = _coerce_gradient(
                record.get("gradient"),
                gradient_path=Path(gradient_path) if gradient_path else None,
            )
            records.append(
                QueryGradient(query_id=query_id, gradient=gradient, metadata=metadata)
            )
    return records


def _load_parquet_gradients(path: Path) -> list[QueryGradient]:
    table = pq.read_table(path)
    records: list[QueryGradient] = []
    for index, row in enumerate(table.to_pylist(), start=1):
        if "query_id" not in row or row["query_id"] is None:
            raise ValueError(
                f"Missing query_id in gradient record at {path}, row {index}"
            )
        query_id = str(row["query_id"])
        gradient_path = row.get("gradient_path")
        metadata = row.get("metadata")
        gradient = _coerce_gradient(
            row.get("gradient"),
            gradient_path=Path(gradient_path) if gradient_path else None,
        )
        records.append(
            QueryGradient(query_id=query_id, gradient=gradient, metadata=metadata)
        )
    return records


def load_query_gradients(path: Path) -> list[QueryGradient]:
    """Load query gradients from a directory, JSONL, or Parquet file."""

    if path.is_dir():
        gradients: list[QueryGradient] = []
        for file in sorted(path.iterdir()):
            if file.suffix.lower() not in {".pt", ".npy"}:
                continue
            gradients.append(
                QueryGradient(query_id=file.stem, gradient=_load_gradient_file(file))
            )
        if not gradients:
            raise FileNotFoundError(f"No gradient files found in {path}")
        return gradients

    if not path.exists():
        raise FileNotFoundError(f"Query gradients path not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _load_jsonl_gradients(path)
    if suffix == ".parquet":
        return _load_parquet_gradients(path)
    raise ValueError(f"Unsupported query gradients format: {path.suffix}")


__all__ = ["load_query_gradients"]
