"""Pure field extraction logic for corpus manifest rows."""

from __future__ import annotations

import re
import urllib.parse
from collections.abc import Mapping

from dolma.constants import FORMATS, TOPICS
from dolma.enrich import extract_url
from dolma.provenance import source_category

TOPIC_INDEX: dict[str, int] = {label: i for i, label in enumerate(TOPICS)}
FORMAT_INDEX: dict[str, int] = {label: i for i, label in enumerate(FORMATS)}

TOPIC_ALIASES: dict[str, str] = {
    "adult": "adult_content",
    "art_design": "art_and_design",
    "crime_law": "crime_and_law",
    "education_jobs": "education_and_jobs",
    "hardware": "electronics_and_hardware",
    "fashion_beauty": "fashion_and_beauty",
    "finance_business": "finance_and_business",
    "food_dining": "food_and_dining",
    "history": "history_and_geography",
    "home_hobbies": "home_and_hobbies",
    "science_tech": "science_math_and_technology",
    "software_dev": "software_development",
    "sports_fitness": "sports_and_fitness",
    "travel": "travel_and_tourism",
}

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_LABEL_PREFIX = "__label__"


def normalize_topic(value: str | None) -> str | None:
    if value is None:
        return None
    return TOPIC_ALIASES.get(value, value)


def strip_label_prefix(value: str | None) -> str | None:
    if value is None:
        return None
    if value.startswith(_LABEL_PREFIX):
        return value[len(_LABEL_PREFIX) :]
    return value


def extract_language(lang_value: object) -> str | None:
    if isinstance(lang_value, str):
        return lang_value
    if isinstance(lang_value, Mapping):
        if not lang_value:
            return None
        return max(lang_value, key=lambda k: lang_value[k])


def compute_bin_id(topic: str | None, format_label: str | None) -> int | None:
    if topic is None or format_label is None:
        return None
    topic_idx = TOPIC_INDEX.get(topic)
    format_idx = FORMAT_INDEX.get(format_label)
    if topic_idx is None or format_idx is None:
        return None
    return topic_idx * len(FORMATS) + format_idx + 1


def bin_id_expr(
    topic_col: str = "weborganizer_topic",
    format_col: str = "weborganizer_format",
):
    """Polars expression form of compute_bin_id (1-indexed, unknown -> null).

    Single source of truth for the bin_id formula over polars frames so EDA /
    bin_manifest paths do not each open-code the index math.
    """
    import polars as pl

    topic_idx = pl.col(topic_col).replace_strict(
        TOPICS, list(range(len(TOPICS))), default=None, return_dtype=pl.Int32
    )
    format_idx = pl.col(format_col).replace_strict(
        FORMATS, list(range(len(FORMATS))), default=None, return_dtype=pl.Int32
    )
    return topic_idx * len(FORMATS) + format_idx + 1


def extract_domain(url: str | None) -> str | None:
    if not url:
        return None
    try:
        hostname = urllib.parse.urlparse(url).hostname
    except ValueError:
        return None
    if not hostname:
        return None
    if hostname.startswith("www."):
        hostname = hostname[4:]
    return hostname


def extract_year(metadata: Mapping[str, object]) -> int | None:
    warc_date = metadata.get("warc_date")
    if not isinstance(warc_date, str):
        return None
    match = _YEAR_RE.search(warc_date)
    if match is None:
        return None
    return int(match.group(0))


def extract_record_row(
    record: Mapping[str, object],
    shard_path: str,
    token_ratio: float,
) -> dict[str, object]:
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}

    raw_topic = metadata.get("weborganizer_topic_max")
    raw_format = metadata.get("weborganizer_format_max")
    topic = normalize_topic(
        strip_label_prefix(raw_topic if isinstance(raw_topic, str) else None)
    )
    format_label = strip_label_prefix(
        raw_format if isinstance(raw_format, str) else None
    )

    word_count = metadata.get("original_word_count")
    estimated_tokens = _extract_token_count(metadata, record.get("text"), token_ratio)

    text = record.get("text")
    char_len = len(text) if isinstance(text, str) else None

    return {
        "doc_id": record.get("id"),
        "shard_path": shard_path,
        "source_family": _extract_source_family(record, metadata, shard_path),
        "weborganizer_topic": topic,
        "weborganizer_format": format_label,
        "bin_id": compute_bin_id(topic, format_label),
        "estimated_token_count": estimated_tokens,
        "token_count": estimated_tokens,
        "document_length_chars": char_len,
        "original_word_count": word_count
        if isinstance(word_count, (int, float))
        else None,
        "language": extract_language(metadata.get("lang")),
        "year": extract_year(metadata),
        "url_domain": extract_domain(extract_url(record)),
    }


def _extract_source_family(
    record: Mapping[str, object],
    metadata: Mapping[str, object],
    shard_path: str,
) -> str:
    soc127 = record.get("_soc_127")
    if isinstance(soc127, Mapping):
        value = soc127.get("source_family")
        if isinstance(value, str) and value:
            return value
    value = metadata.get("source_family")
    if isinstance(value, str) and value:
        return value
    return source_category(shard_path)


def _extract_token_count(
    metadata: Mapping[str, object],
    text: object,
    token_ratio: float,
) -> int | None:
    token_count_est = metadata.get("token_count_est")
    if isinstance(token_count_est, (int, float)):
        return int(token_count_est)

    word_count = metadata.get("original_word_count")
    if isinstance(word_count, (int, float)):
        return max(1, int(word_count * token_ratio + 0.5))

    if isinstance(text, str):
        from dolma.sample import approximate_token_count

        return approximate_token_count(text, metadata)
    return None


__all__ = [
    "FORMAT_INDEX",
    "TOPIC_ALIASES",
    "TOPIC_INDEX",
    "bin_id_expr",
    "compute_bin_id",
    "extract_domain",
    "extract_language",
    "extract_record_row",
    "extract_year",
    "normalize_topic",
    "strip_label_prefix",
]
