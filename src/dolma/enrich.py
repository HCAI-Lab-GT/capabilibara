"""Record enrichment helpers."""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
from urllib.parse import urlparse

from .constants import WORD_TO_TOKEN_MULTIPLIER
from .sample import approximate_token_count


def _compute_word_count(text: object) -> int | None:
    if not isinstance(text, str):
        return None
    return len(text.split()) if text else 0


def extract_url(record: Mapping[str, object]) -> str | None:
    metadata = record.get("metadata")
    if isinstance(metadata, Mapping):
        value = metadata.get("warc_url")
        if isinstance(value, str) and value:
            return value
        value = metadata.get("url")
        if isinstance(value, str) and value:
            return value
        value = metadata.get("source_url")
        if isinstance(value, str) and value:
            return value
    return None


def extract_domain(url: str | None) -> str | None:
    if not url:
        return None
    try:
        hostname = urlparse(url).hostname
    except ValueError:
        return None
    return hostname or None


def estimate_tokens(text: str, metadata: Mapping[str, object]) -> int:
    word_count = metadata.get("original_word_count")
    if isinstance(word_count, (int, float)) and word_count > 0:
        return max(1, int(word_count * WORD_TO_TOKEN_MULTIPLIER + 0.5))
    return approximate_token_count(text, metadata)


def enrich_record(
    record: Mapping[str, object],
    format_probs: Mapping[str, float],
    format_max: str,
    topic_probs: Mapping[str, float] | None,
    topic_max: str | None,
    *,
    fill_word_count: bool = False,
) -> dict[str, object]:
    """Attach topic/format predictions and fill word count if missing."""

    enriched = deepcopy(record)
    metadata = enriched.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        enriched["metadata"] = metadata

    # Migrate legacy topic keys if present.
    if "weborganizer" in metadata:
        metadata["weborganizer_topic"] = metadata.get(
            "weborganizer_topic", metadata["weborganizer"]
        )
        metadata.pop("weborganizer", None)
    if "weborganizer_max" in metadata:
        metadata["weborganizer_topic_max"] = metadata.get(
            "weborganizer_topic_max", metadata["weborganizer_max"]
        )
        metadata.pop("weborganizer_max", None)

    if topic_probs is not None and topic_max is not None:
        metadata["weborganizer_topic"] = dict(topic_probs)
        metadata["weborganizer_topic_max"] = topic_max

    metadata["weborganizer_format"] = dict(format_probs)
    metadata["weborganizer_format_max"] = format_max

    if fill_word_count and "original_word_count" not in metadata:
        wc = _compute_word_count(enriched.get("text"))
        if wc is not None:
            metadata["original_word_count"] = wc

    text = enriched.get("text") or ""
    if not isinstance(text, str):
        text = ""
    metadata["token_count_est"] = estimate_tokens(text, metadata)
    metadata["char_length"] = len(text)
    metadata["domain"] = extract_domain(extract_url(enriched))

    return enriched


__all__ = [
    "enrich_record",
    "estimate_tokens",
    "extract_domain",
    "extract_url",
]
