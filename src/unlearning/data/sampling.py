"""Stratified sampling for unlearning forget and retain sets.

Handles:
  - document length filtering
  - forget set sampling with manifest
  - stratified retain sampling from non-target topics

Token count estimation uses a word-count heuristic to avoid tokenizing
the full ~5.5M-row Arrow cache.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

logger = logging.getLogger(__name__)

TOPIC_COL = "weborganizer_topic"
TEXT_COL = "text"
DOC_ID_COL = "doc_id"
WORD_TO_TOKEN_MULTIPLIER = 1.35


def filter_num_proc() -> int | None:
    """Return optional HF Datasets multiprocessing setting for filters."""
    raw_value = os.environ.get("UNLEARNING_FILTER_NUM_PROC")
    if raw_value is None or raw_value == "":
        return None
    value = int(raw_value)
    return value if value > 1 else None


def estimate_token_count(text: str) -> float:
    return len(text.split()) * WORD_TO_TOKEN_MULTIPLIER


def filter_by_min_tokens(
    ds,
    min_tokens: int = 512,
    topic_col: str = TOPIC_COL,
) -> tuple:
    """Filter dataset to documents with estimated token count >= min_tokens.

    Returns (filtered_ds, coverage_stats) where coverage_stats maps
    topic -> {"before": int, "after": int}.
    """
    from collections import Counter

    before_counts = Counter(ds[topic_col])

    min_words = int(min_tokens / WORD_TO_TOKEN_MULTIPLIER)
    filtered = ds.filter(
        lambda x: len(x[TEXT_COL].split()) >= min_words,
        num_proc=filter_num_proc(),
        desc=f"filter:min_tokens>={min_tokens}",
    )

    after_counts = Counter(filtered[topic_col])
    coverage: dict[str, dict[str, int]] = {}
    for topic in before_counts:
        b = before_counts[topic]
        a = after_counts.get(topic, 0)
        coverage[topic] = {"before": b, "after": a}
        if a < b:
            logger.info(
                "  %s: %d -> %d docs (%.1f%% retained)",
                topic,
                b,
                a,
                100.0 * a / b if b else 0,
            )

    logger.info(
        "Min-token filter (%d): %d -> %d docs total",
        min_tokens,
        len(ds),
        len(filtered),
    )
    return filtered, coverage


def sample_forget(
    ds,
    target_topics: list[str],
    max_docs: int,
    rng: random.Random,
    topic_col: str = TOPIC_COL,
    text_col: str = TEXT_COL,
    doc_id_col: str = DOC_ID_COL,
) -> tuple[list[str], list[str]]:
    """Sample forget documents from target topic(s).

    Returns (texts, doc_ids). Stratified across target_topics when len > 1.
    """
    if len(target_topics) == 1:
        topic = target_topics[0]
        topic_ds = ds.filter(
            lambda x: x[topic_col] == topic,
            num_proc=filter_num_proc(),
            desc=f"filter:forget:{topic}",
        )
        n = min(max_docs, len(topic_ds))
        indices = list(range(len(topic_ds)))
        rng.shuffle(indices)
        selected = topic_ds.select(indices[:n])
        texts = selected[text_col]
        doc_ids = selected[doc_id_col]
        logger.info("Forget: topic='%s', %d/%d docs sampled", topic, n, len(topic_ds))
    else:
        per_topic = max(1, max_docs // len(target_topics))
        texts, doc_ids = [], []
        for topic in target_topics:
            topic_ds = ds.filter(
                lambda x, t=topic: x[topic_col] == t,
                num_proc=filter_num_proc(),
                desc=f"filter:forget:{topic}",
            )
            n = min(per_topic, len(topic_ds))
            indices = list(range(len(topic_ds)))
            rng.shuffle(indices)
            selected = topic_ds.select(indices[:n])
            texts.extend(selected[text_col])
            doc_ids.extend(selected[doc_id_col])
            logger.info("Forget: topic='%s', %d/%d docs", topic, n, len(topic_ds))

    if len(texts) < max_docs:
        logger.warning(
            "Forget shortfall: requested %d, got %d",
            max_docs,
            len(texts),
        )
    texts = [t for t in texts if t and t.strip()]
    return texts, doc_ids


def sample_retain_stratified(
    ds,
    exclude_topics: set[str],
    docs_per_bin: int,
    rng: random.Random,
    all_topics: list[str] | None = None,
    topic_col: str = TOPIC_COL,
    text_col: str = TEXT_COL,
    doc_id_col: str = DOC_ID_COL,
) -> tuple[list[str], list[str], dict[str, int]]:
    """Sample docs_per_bin from each non-target topic.

    Returns (texts, doc_ids, per_topic_counts).
    """
    if all_topics is None:
        from dolma.constants import TOPICS

        all_topics = TOPICS

    retain_topics = [t for t in all_topics if t not in exclude_topics]
    texts, doc_ids = [], []
    per_topic_counts: dict[str, int] = {}

    for topic in retain_topics:
        topic_ds = ds.filter(
            lambda x, t=topic: x[topic_col] == t,
            num_proc=filter_num_proc(),
            desc=f"filter:retain:{topic}",
        )
        n = min(docs_per_bin, len(topic_ds))
        if n < docs_per_bin:
            logger.warning(
                "Retain bin '%s': only %d docs available (requested %d)",
                topic,
                len(topic_ds),
                docs_per_bin,
            )
        indices = list(range(len(topic_ds)))
        rng.shuffle(indices)
        selected = topic_ds.select(indices[:n])
        texts.extend(selected[text_col])
        doc_ids.extend(selected[doc_id_col])
        per_topic_counts[topic] = n

    texts = [t for t in texts if t and t.strip()]
    total = sum(per_topic_counts.values())
    logger.info(
        "Retain: %d topics x ~%d docs = %d total",
        len(retain_topics),
        docs_per_bin,
        total,
    )
    return texts, doc_ids, per_topic_counts


def save_sampling_manifest(
    output_dir: str,
    forget_doc_ids: list[str],
    retain_doc_ids: list[str],
    seed: int,
    target_topics: list[str],
    config_snapshot: dict,
    forget_text_count: int | None = None,
    retain_text_count: int | None = None,
) -> Path:
    """Write sampling_manifest.json to output_dir."""
    path = Path(output_dir) / "sampling_manifest.json"
    manifest = {
        "seed": seed,
        "target_topics": target_topics,
        "forget_count": len(forget_doc_ids),
        "retain_count": len(retain_doc_ids),
        "forget_text_count": forget_text_count,
        "retain_text_count": retain_text_count,
        "forget_doc_ids": forget_doc_ids,
        "retain_doc_ids": retain_doc_ids,
        "config": config_snapshot,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Sampling manifest saved to %s", path)
    return path
