"""Convert a working sample manifest into EDA-format CSVs.

The existing report pipeline (data_loader.py) expects three CSVs with
word-count-based columns produced by the EDA aggregation over raw JSONL
shards.  Working sample manifests have token_count instead.  This module
bridges the gap by back-converting tokens to estimated word counts so
the downstream round-trip (word_count_sum * 1.35 -> token_count_est)
recovers the original token counts.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from dolma.constants import WORD_TO_TOKEN_MULTIPLIER
from dolma.distribution_report.sampling_loader import normalize_taxonomy_label

logger = logging.getLogger(__name__)

_EDA_FILENAMES = {
    "topic": "topic_label_counts.csv",
    "format": "format_label_counts.csv",
    "joint": "topic_format_joint_counts.csv",
}


def _normalize_label(raw: str) -> str:
    return normalize_taxonomy_label(raw)


def _load_manifest(manifest_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(manifest_path)

    topic_col = next(
        (c for c in ("bin_topic", "weborganizer_topic", "topic") if c in df.columns),
        None,
    )
    format_col = next(
        (c for c in ("bin_format", "weborganizer_format", "format") if c in df.columns),
        None,
    )
    if topic_col is None or format_col is None:
        raise ValueError(
            f"Manifest must contain topic and format columns.  Found: {list(df.columns)}"
        )

    token_col = next(
        (c for c in ("token_count", "estimated_token_count") if c in df.columns),
        None,
    )
    if token_col is None:
        raise ValueError(
            f"Manifest must contain a token count column.  Found: {list(df.columns)}"
        )

    out = pd.DataFrame(
        {
            "topic_label": df[topic_col].map(_normalize_label),
            "format_label": df[format_col].map(_normalize_label),
            "token_count": pd.to_numeric(df[token_col], errors="raise").astype(int),
        }
    )
    return out


def _build_marginal(
    df: pd.DataFrame,
    group_col: str,
    multiplier: float,
) -> pd.DataFrame:
    grouped = df.groupby(group_col, as_index=False).agg(
        doc_count=("token_count", "size"),
        token_sum=("token_count", "sum"),
    )
    grouped = grouped.rename(columns={group_col: "label"})
    grouped["word_count_sum"] = (grouped["token_sum"] / multiplier).astype(int)
    grouped["word_count_docs"] = grouped["doc_count"]
    grouped["word_count_mean"] = grouped["word_count_sum"] / grouped["word_count_docs"]
    grouped["score_sum"] = grouped["doc_count"].astype(float)
    grouped["score_mean"] = 1.0
    return grouped[
        [
            "label",
            "doc_count",
            "score_sum",
            "score_mean",
            "word_count_sum",
            "word_count_docs",
            "word_count_mean",
        ]
    ]


def _build_joint(df: pd.DataFrame, multiplier: float) -> pd.DataFrame:
    grouped = df.groupby(["topic_label", "format_label"], as_index=False).agg(
        doc_count=("token_count", "size"),
        token_sum=("token_count", "sum"),
    )
    grouped["word_count_sum"] = (grouped["token_sum"] / multiplier).astype(int)
    grouped["word_count_docs"] = grouped["doc_count"]
    grouped["word_count_mean"] = grouped["word_count_sum"] / grouped["word_count_docs"]
    return grouped[
        [
            "topic_label",
            "format_label",
            "doc_count",
            "word_count_sum",
            "word_count_docs",
            "word_count_mean",
        ]
    ]


def manifest_to_eda_csvs(
    manifest_path: Path,
    output_dir: Path,
    *,
    multiplier: float = WORD_TO_TOKEN_MULTIPLIER,
) -> dict[str, Path]:
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _load_manifest(manifest_path)
    logger.info(
        "Loaded manifest: %s docs, %s tokens",
        f"{len(df):,}",
        f"{df['token_count'].sum():,}",
    )

    topic_df = _build_marginal(df, "topic_label", multiplier)
    format_df = _build_marginal(df, "format_label", multiplier)
    joint_df = _build_joint(df, multiplier)

    paths: dict[str, Path] = {}
    for key, filename in _EDA_FILENAMES.items():
        out_path = output_dir / filename
        frame = {"topic": topic_df, "format": format_df, "joint": joint_df}[key]
        frame.to_csv(out_path, index=False)
        logger.info("Wrote %s (%d rows)", out_path, len(frame))
        paths[key] = out_path

    return paths


__all__ = ["manifest_to_eda_csvs"]
