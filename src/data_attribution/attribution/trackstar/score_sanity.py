"""Sanity-check helpers for TrackStar scores.

Two pre-score guards with deliberately different strictness:
``assert_module_space_match`` keeps the legacy intersection semantics for the
directory-scan scoring path (partial module overlap scores the shared keys),
while ``assert_gradient_tensor_space_match`` is the controlled-path guard that
requires the exact same module set and validates tensor structure.
``nonzero_fraction`` is an opt-in post-score guard for all-zero output
directories.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch


def assert_module_space_match(
    query_grad_keys: set[str], index_grad_keys: set[str]
) -> set[str]:
    shared = query_grad_keys & index_grad_keys
    if not shared:
        raise ValueError(
            "Query and index gradient module keys are disjoint; scoring would "
            f"return all-zeros. query={sorted(query_grad_keys)} "
            f"index={sorted(index_grad_keys)}"
        )
    return shared


def nonzero_fraction(npy_dir: Path) -> float:
    total = 0
    nonzero = 0
    for npy_path in sorted(npy_dir.glob("shard_*.npy")):
        arr = np.load(npy_path, mmap_mode="r")
        total += int(arr.size)
        nonzero += int(np.count_nonzero(arr))
    if total == 0:
        return 0.0
    return float(nonzero / total)


def _assert_tensor_structure(
    grads: Mapping[str, torch.Tensor], side: str, keys: tuple[str, ...]
) -> None:
    expected_rows: int | None = None
    for key in keys:
        tensor = grads[key]
        if tensor.ndim != 2:
            raise ValueError(
                f"{side} gradient tensor for {key} must be two-dimensional; "
                f"got shape {tuple(tensor.shape)}"
            )
        width = int(tensor.shape[1])
        if width < 1:
            raise ValueError(
                f"{side} gradient projected width for {key} must be positive; "
                f"got {width}"
            )
        rows = int(tensor.shape[0])
        if expected_rows is not None and rows != expected_rows:
            raise ValueError(
                f"{side} gradient row count mismatch for {key}: "
                f"expected {expected_rows}, got {rows}"
            )
        expected_rows = rows


def assert_gradient_tensor_space_match(
    query_grads: Mapping[str, torch.Tensor],
    document_grads: Mapping[str, torch.Tensor],
) -> tuple[str, ...]:
    query_keys = set(query_grads)
    document_keys = set(document_grads)
    if query_keys != document_keys:
        raise ValueError(
            "Query and document gradient module keys must match exactly; "
            f"query_only={sorted(query_keys - document_keys)} "
            f"document_only={sorted(document_keys - query_keys)}"
        )
    keys = tuple(sorted(query_keys))
    if not keys:
        raise ValueError("Query and document gradient module mappings cannot be empty")
    _assert_tensor_structure(query_grads, "query", keys)
    _assert_tensor_structure(document_grads, "document", keys)
    for key in keys:
        query_width = int(query_grads[key].shape[1])
        document_width = int(document_grads[key].shape[1])
        if query_width != document_width:
            raise ValueError(
                f"Gradient projected width mismatch for {key}: "
                f"query={query_width}, document={document_width}"
            )
    return keys
