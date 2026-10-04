"""Sampling helpers for Dolma records."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

from .constants import DOLMA_FIELDS


def ensure_metadata(record: Mapping[str, object]) -> dict[str, object]:
    metadata = record.get("metadata")
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise ValueError("Dolma metadata must be a mapping when provided.")
    return dict(metadata)


def project_record(record: Mapping[str, object]) -> dict[str, object]:
    return {field: record.get(field) for field in DOLMA_FIELDS[:-1]} | {
        "metadata": ensure_metadata(record)
    }


def approximate_token_count(text: str, metadata: Mapping[str, object]) -> int:
    tokens = metadata.get("num_tokens")
    if isinstance(tokens, int):
        return tokens
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def _constraints_met(
    document_count: int,
    token_total: int,
    num_documents: int | None,
    token_budget: int | None,
) -> bool:
    if token_budget is not None and token_total >= token_budget:
        return True
    return num_documents is not None and document_count >= num_documents


def _stratum_key(
    record: Mapping[str, object], stratify_by: Sequence[str]
) -> tuple[object, ...]:
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        return tuple(None for _ in stratify_by)
    return tuple(metadata.get(field) for field in stratify_by)


def sample_documents(
    records: Iterable[Mapping[str, object]],
    *,
    num_documents: int | None = None,
    token_budget: int | None = None,
    stratify_by: Sequence[str] | None = None,
) -> tuple[list[dict[str, object]], int]:
    """Sample documents from a stream, optionally using greedy stratification.

    .. warning::
       Stratification is greedy and order-dependent. It assumes the input stream is mixed.
       If the input is clustered by stratum, earlier strata will fill the quota first.
       This is an approximate method, not a guaranteed representative sample.
    """
    if num_documents is None and token_budget is None:
        raise ValueError("Provide num_documents or token_budget to control sampling")

    stratify_fields = list(stratify_by or [])
    selected: list[dict[str, object]] = []
    token_total = 0
    strata_counts: dict[tuple[object, ...], int] = {}

    for record in records:
        projected = project_record(record)
        tokens = approximate_token_count(projected["text"] or "", projected["metadata"])

        stratum_key = (
            _stratum_key(projected, stratify_fields) if stratify_fields else None
        )
        if stratify_fields:
            existing_strata = len(strata_counts) + (
                0 if stratum_key in strata_counts else 1
            )
            if num_documents is not None:
                target = math.ceil(num_documents / max(1, existing_strata))
                if strata_counts.get(stratum_key, 0) >= target:
                    continue
        selected.append(projected)
        token_total += tokens
        if stratify_fields and stratum_key is not None:
            strata_counts[stratum_key] = strata_counts.get(stratum_key, 0) + 1

        if _constraints_met(len(selected), token_total, num_documents, token_budget):
            break

    return selected, token_total


__all__ = [
    "approximate_token_count",
    "ensure_metadata",
    "project_record",
    "sample_documents",
]
