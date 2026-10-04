"""Index helpers for attribution runs."""

from __future__ import annotations

import importlib
import importlib.util
import logging
from pathlib import Path
from collections.abc import Mapping

import torch

from data_attribution.attribution.scoring import _ensure_device


def _is_bergson_index(path: Path) -> bool:
    return path.is_dir() and (path / "gradients.bin").exists()


def _load_index_ids(
    path: Path, field: str, logger: logging.Logger
) -> list[object] | None:
    data_path = path / "data.hf"
    if not data_path.exists():
        logger.debug("No data.hf found under %s; using positional ids", path)
        return None

    if importlib.util.find_spec("datasets") is None:
        logger.debug("datasets package not available; falling back to positional ids")
        return None

    datasets_module = importlib.import_module("datasets")
    try:
        dataset = datasets_module.load_from_disk(str(data_path))
    except Exception as exc:
        logger.debug("Failed to load %s: %s", data_path, exc)
        return None
    split = next(iter(dataset.values())) if isinstance(dataset, dict) else dataset
    if field not in split.column_names:
        logger.debug(
            "Column %s not present in %s; using positional ids", field, data_path
        )
        return None

    return [row[field] for row in split]


def _import_attributor():
    spec = importlib.util.find_spec("bergson.query.attributor")
    if spec is None:
        raise RuntimeError("Install the bergson package to run attribution.")
    module = importlib.import_module("bergson.query.attributor")
    return module.Attributor


def _load_index_gradients(
    path: Path,
    *,
    device: str | None,
    unit_norm: bool,
) -> tuple[object, Mapping[str, torch.Tensor]]:
    Attributor = _import_attributor()
    attributor = Attributor(index_path=path, device=device, unit_norm=unit_norm)
    grads = {
        name: _ensure_device(tensor, device)
        for name, tensor in attributor.grads.items()
    }
    if not grads:
        raise ValueError(f"No gradients found under {path}")
    return attributor, grads


__all__ = [
    "_import_attributor",
    "_is_bergson_index",
    "_load_index_gradients",
    "_load_index_ids",
]
