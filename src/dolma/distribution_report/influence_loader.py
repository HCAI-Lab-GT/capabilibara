"""Load bin-level influence aggregation CSVs for heatmap rendering."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from dolma.distribution_report.sampling_loader import (
    CANONICAL_FORMAT_LABELS,
    CANONICAL_TOPIC_LABELS,
)

logger = logging.getLogger(__name__)

BENCHMARK_DISPLAY_NAMES: dict[str, str] = {
    "queries_gsm8k": "GSM8K",
    "queries_socialiqa": "SocialIQA",
    "queries_mmlu_social_science": "MMLU Social Sciences",
    "queries_mmlu_stem": "MMLU STEM",
    "queries_arc_easy": "ARC-Easy",
    "queries_arc_challenge": "ARC-Challenge",
    "queries_bbh_snarks": "BBH Snarks",
    "queries_bbh_causal_judgement": "BBH Causal Judgement",
    "queries_bbh_sports_understanding": "BBH Sports Understanding",
}

_LABEL_PREFIX = "__label__"


def _normalize_label(label: str) -> str:
    label = label.strip().lower().replace(" ", "_")
    if not label.startswith(_LABEL_PREFIX):
        return f"{_LABEL_PREFIX}{label}"
    return label


_PERQUERY_COLUMN_MAP = {
    "median_influence": "mean_score",
    "median_abs_influence": "abs_mean_score",
    "std_influence": "std_score",
}


def _apply_column_map(df: pd.DataFrame) -> pd.DataFrame:
    if "median_influence" in df.columns:
        df = df.rename(columns=_PERQUERY_COLUMN_MAP)
        logger.debug("Mapped perquery columns to standard names")
    return df


def _z_score_in_place(df: pd.DataFrame, column: str) -> None:
    """Z-score `column` within the dataframe (= within one benchmark).

    Replaces raw signed-influence values (~1e-5 magnitude) with
    standardized z-scores so downstream colorbars and prose are
    consistent with the manuscript's "z-score units" semantics
    (e.g., Finding 1's "$\\Delta z = +13.1$").

    Safe no-op when std is 0 or column is missing.
    """
    if column not in df.columns:
        return
    series = df[column]
    mu = float(series.mean())
    sigma = float(series.std())
    if sigma == 0 or not np.isfinite(sigma):
        df[column] = 0.0
        return
    df[column] = (series - mu) / sigma


def load_influence_grid(csv_path: Path, z_score: bool = True) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["topic_label"] = df["topic_label"].apply(_normalize_label)
    df["format_label"] = df["format_label"].apply(_normalize_label)
    df = _apply_column_map(df)
    if z_score:
        # Standardize the signed-influence column within this benchmark
        # so the "z-score" axis label is honest and colorbar ticks are
        # readable in the O(1) range instead of collapsing to 0.00 at the
        # raw ~1e-5 magnitude.
        _z_score_in_place(df, "mean_score")
    return df


def load_all_influence_grids(
    influence_dir: Path,
) -> dict[str, pd.DataFrame]:
    grids: dict[str, pd.DataFrame] = {}
    for benchmark_key in BENCHMARK_DISPLAY_NAMES:
        perquery_path = influence_dir / f"{benchmark_key}_bin_scores_perquery.csv"
        legacy_path = influence_dir / f"{benchmark_key}_bin_scores.csv"
        csv_path = perquery_path if perquery_path.exists() else legacy_path
        if not csv_path.exists():
            logger.warning("Missing: %s", legacy_path)
            continue
        grids[benchmark_key] = load_influence_grid(csv_path)
        logger.info("Loaded %s: %d rows", benchmark_key, len(grids[benchmark_key]))
    return grids


def influence_axis_order(
    grid_df: pd.DataFrame,
    value_col: str = "abs_mean_score",
) -> tuple[list[str], list[str]]:
    topic_totals = (
        grid_df.groupby("topic_label")[value_col].sum().sort_values(ascending=False)
    )
    format_totals = (
        grid_df.groupby("format_label")[value_col].sum().sort_values(ascending=False)
    )
    topic_order = [t for t in topic_totals.index if t in set(CANONICAL_TOPIC_LABELS)]
    format_order = [f for f in format_totals.index if f in set(CANONICAL_FORMAT_LABELS)]
    return topic_order, format_order


def canonical_axis_order() -> tuple[list[str], list[str]]:
    return list(CANONICAL_TOPIC_LABELS), list(CANONICAL_FORMAT_LABELS)


__all__ = [
    "BENCHMARK_DISPLAY_NAMES",
    "canonical_axis_order",
    "influence_axis_order",
    "load_all_influence_grids",
    "load_influence_grid",
]
