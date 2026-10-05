"""Shared sampling operations for manifest-backed corpus sampling."""

from __future__ import annotations

import random
import uuid

import pandas as pd

from dolma.constants import (
    BIN_EMPTY,
    BIN_FULL,
    BIN_PARTIAL,
    BIN_SPARSE,
    FORMATS,
    TARGET_DOCS_PER_BIN,
    TOKEN_FLOOR_PER_BIN,
    TOPICS,
)
from dolma.manifest_fields import compute_bin_id

SEED = 42
REPRESENTATIVE_TOKEN_TARGET = 5_000_000_000


def generate_dummy_manifest(n_docs: int = 500_000, seed: int = SEED) -> pd.DataFrame:
    rng = random.Random(seed)
    topic_weights = [rng.uniform(0.5, 2.0) for _ in TOPICS]
    topic_weights[TOPICS.index("science_math_and_technology")] = 80.0
    topic_weights[TOPICS.index("software_development")] = 60.0
    topic_weights[TOPICS.index("entertainment")] = 50.0
    topic_weights[TOPICS.index("health")] = 45.0
    topic_weights[TOPICS.index("software")] = 40.0
    format_weights = [rng.uniform(0.5, 2.0) for _ in FORMATS]
    format_weights[FORMATS.index("product_page")] = 70.0
    format_weights[FORMATS.index("nonfiction_writing")] = 65.0
    format_weights[FORMATS.index("news_article")] = 60.0
    format_weights[FORMATS.index("knowledge_article")] = 55.0
    format_weights[FORMATS.index("personal_blog")] = 50.0
    records = []
    for _ in range(n_docs):
        topic = rng.choices(TOPICS, weights=topic_weights, k=1)[0]
        format_label = rng.choices(FORMATS, weights=format_weights, k=1)[0]
        word_count = int(rng.lognormvariate(5.5, 1.2))
        records.append(
            {
                "doc_id": str(uuid.uuid4()),
                "shard_path": f"data/common_crawl-{topic}-{rng.randint(10, 19):04d}/shard_{rng.randint(0, 99):08d}.jsonl.zst",
                "token_count": int(word_count * 1.35),
                "topic": topic,
                "format": format_label,
                "bin_id": compute_bin_id(topic, format_label),
            }
        )
    return pd.DataFrame(records)


def compute_bin_stats(
    manifest: pd.DataFrame,
    target_docs_per_bin: int = TARGET_DOCS_PER_BIN,
) -> pd.DataFrame:
    stats = []
    for topic in TOPICS:
        for format_label in FORMATS:
            bin_df = manifest[
                (manifest["topic"] == topic) & (manifest["format"] == format_label)
            ]
            doc_count = len(bin_df)
            token_count = int(bin_df["token_count"].sum())
            mean_tokens = float(bin_df["token_count"].mean()) if doc_count > 0 else 0.0
            median_tokens = (
                float(bin_df["token_count"].median()) if doc_count > 0 else 0.0
            )
            if doc_count == 0:
                classification = BIN_EMPTY
            elif doc_count < 1_000:
                classification = BIN_SPARSE
            elif doc_count < target_docs_per_bin:
                classification = BIN_PARTIAL
            else:
                classification = BIN_FULL
            stats.append(
                {
                    "bin_id": compute_bin_id(topic, format_label),
                    "topic": topic,
                    "format": format_label,
                    "doc_count": doc_count,
                    "token_count": token_count,
                    "mean_tokens_per_doc": round(mean_tokens, 1),
                    "median_tokens_per_doc": round(median_tokens, 1),
                    "classification": classification,
                }
            )
    return pd.DataFrame(stats)


def stratified_sample(
    manifest: pd.DataFrame,
    bin_stats: pd.DataFrame,
    seed: int = SEED,
    target_docs_per_bin: int = TARGET_DOCS_PER_BIN,
) -> pd.DataFrame:
    rng = random.Random(seed)
    sampled_parts = []
    for _, bin_row in bin_stats.iterrows():
        if bin_row["classification"] == BIN_EMPTY:
            continue
        bin_df = manifest[
            (manifest["topic"] == bin_row["topic"])
            & (manifest["format"] == bin_row["format"])
        ].copy()
        if bin_row["classification"] == BIN_FULL:
            sample = bin_df.sample(
                n=target_docs_per_bin, random_state=rng.randint(0, 2**31)
            )
            if sample["token_count"].sum() < TOKEN_FLOOR_PER_BIN:
                remaining = bin_df[~bin_df["doc_id"].isin(sample["doc_id"])]
                extra = remaining.sample(
                    n=min(len(remaining), target_docs_per_bin),
                    random_state=rng.randint(0, 2**31),
                )
                sample = pd.concat([sample, extra], ignore_index=True)
        else:
            sample = bin_df
        sample = sample.copy()
        sample["bin_id"] = sample["bin_id"].fillna(int(bin_row["bin_id"]))
        sample["bin_classification"] = bin_row["classification"]
        sample["sample"] = "stratified"
        sampled_parts.append(sample)
    return pd.concat(sampled_parts, ignore_index=True)


def representative_sample(
    manifest: pd.DataFrame,
    stratified_df: pd.DataFrame,
    token_target: int = REPRESENTATIVE_TOKEN_TARGET,
    seed: int = SEED,
) -> pd.DataFrame:
    rng = random.Random(seed)
    sampled_ids = set(stratified_df["doc_id"])
    remaining = manifest[~manifest["doc_id"].isin(sampled_ids)].copy()
    if remaining.empty:
        empty = remaining.copy()
        empty["sample"] = "representative"
        return empty
    remaining["_sampling_bin_key"] = remaining["topic"] + "__" + remaining["format"]
    total_tokens = remaining["token_count"].sum()
    bin_token_shares = (
        remaining.groupby("_sampling_bin_key")["token_count"].sum() / total_tokens
    )
    parts = []
    for bin_key, share in bin_token_shares.items():
        bin_df = remaining[remaining["_sampling_bin_key"] == bin_key].copy()
        bin_target_tokens = int(token_target * share)
        bin_df = bin_df.sample(frac=1, random_state=rng.randint(0, 2**31))
        bin_df["cumulative_tokens"] = bin_df["token_count"].cumsum()
        sampled = bin_df[bin_df["cumulative_tokens"] <= bin_target_tokens].copy()
        sampled = sampled.drop(columns=["_sampling_bin_key", "cumulative_tokens"])
        parts.append(sampled)
    rep_df = pd.concat(parts, ignore_index=True)
    rep_df["sample"] = "representative"
    return rep_df


__all__ = [
    "REPRESENTATIVE_TOKEN_TARGET",
    "SEED",
    "compute_bin_stats",
    "generate_dummy_manifest",
    "representative_sample",
    "stratified_sample",
]
