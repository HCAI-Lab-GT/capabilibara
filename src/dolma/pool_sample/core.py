"""Sampling helpers for Dolma pool records."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

from dolma.constants import WORD_TO_TOKEN_MULTIPLIER
from dolma.sample import approximate_token_count, project_record


def estimate_pool_tokens(text: str, metadata: Mapping[str, object]) -> int:
    word_count = metadata.get("original_word_count")
    if isinstance(word_count, (int, float)) and word_count > 0:
        return max(1, math.ceil(word_count * WORD_TO_TOKEN_MULTIPLIER))
    return approximate_token_count(text, metadata)


def stream_sample(
    records: Iterable[Mapping[str, object]],
    token_budget: int,
) -> Iterable[tuple[dict[str, object], int]]:
    """Yield (projected_record, cumulative_tokens) until budget met."""
    cumulative = 0
    for record in records:
        projected = project_record(record)
        text = projected.get("text") or ""
        tokens = estimate_pool_tokens(text, projected.get("metadata") or {})
        cumulative += tokens
        yield projected, cumulative
        if cumulative >= token_budget:
            return


def extract_manifest_row(
    record: Mapping[str, object],
    token_count: int,
    shard_path: str,
) -> dict[str, object]:
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}
    return {
        "doc_id": record.get("id"),
        "shard_path": shard_path,
        "token_count": token_count,
        "weborganizer_topic": metadata.get("weborganizer_topic"),
        "weborganizer_format": metadata.get("weborganizer_format"),
    }


__all__ = [
    "estimate_pool_tokens",
    "extract_manifest_row",
    "stream_sample",
]
