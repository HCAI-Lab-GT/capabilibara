"""Load and prepare EDA aggregates for WebOrganizer report figures."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from dolma.constants import WORD_TO_TOKEN_MULTIPLIER
from dolma.distribution_report.sampling_loader import (
    CANONICAL_FORMAT_LABELS,
    CANONICAL_LABEL,
    CANONICAL_TOPIC_LABELS,
    TYPO_LABEL,
)

_MARGINAL_SUM_COLS = ("doc_count", "score_sum", "word_count_sum", "word_count_docs")
_JOINT_SUM_COLS = ("doc_count", "word_count_sum", "word_count_docs")
_EDA_REQUIRED_FILES = {
    "topic": "topic_label_counts.csv",
    "format": "format_label_counts.csv",
    "joint": "topic_format_joint_counts.csv",
}


def strip_label_prefix(label: str) -> str:
    return label.removeprefix("__label__").replace("_", " ")


_DISPLAY_OVERRIDES: dict[str, str] = {
    "faq": "FAQ",
    "q a forum": "Q&A Forum",
    "about org": "About Org",
    "about pers": "About Person",
    "news org": "News Org",
}


# Canonical short-label mapping for space-constrained figures (Fig 4 panel a,
# heatmap axis ticks at narrow column widths, etc.). Keys are the
# raw-but-spaced form returned by ``strip_label_prefix``; values are the
# preferred short form. Topics + formats are mixed in one namespace because
# their label spaces don't collide.
#
# Maintenance: when adding a new topic/format to the WebOrganizer taxonomy,
# add its short form here. If a label has no entry, ``short_label`` falls
# back to ``display_label`` (the full Title Case form).
_SHORT_OVERRIDES: dict[str, str] = {
    # Topics
    "adult content":              "Adult",
    "art and design":             "Arts",
    "crime and law":              "Crime",
    "education and jobs":         "Education",
    "electronics and hardware":   "Electronics",
    "electronics and hardare":    "Electronics",  # historical typo in some EDA exports
    "fashion and beauty":         "Fashion",
    "finance and business":       "Finance",
    "food and dining":            "Food",
    "history and geography":      "History",
    "home and hobbies":           "Hobbies",
    "science math and technology": "Sci. & Tech.",
    "social life":                "Social",
    "software development":       "Software Dev.",
    "sports and fitness":         "Sports",
    "transportation":             "Transport",
    "travel and tourism":         "Travel",
    # Formats (keep _DISPLAY_OVERRIDES forms for FAQ/Q&A/Org/Pers since
    # they're already short; add abbreviations for long ones).
    "academic writing":           "Academic",
    "audio transcript":           "Transcript",
    "comment section":            "Comments",
    "content listing":            "Listing",
    "creative writing":           "Creative",
    "customer support":           "Support",
    "documentation":              "Docs",
    "knowledge article":          "Knowledge",
    "legal notices":              "Legal",
    "news article":               "News",
    "nonfiction writing":         "Nonfiction",
    "personal blog":              "Blog",
    "product page":               "Product",
    "spam ads":                   "Ads",
    "structured data":            "Data",
    "user review":                "Review",
}


def display_label(label: str) -> str:
    raw = strip_label_prefix(label)
    return _DISPLAY_OVERRIDES.get(raw, raw.title())


def short_label(label: str) -> str:
    """Compact display label for space-constrained figures.

    Use in figures where ``display_label`` would overflow the available
    space (e.g., Fig 4 panel a with 24 y-axis ticks, heatmap format ticks
    at narrow COLM column width). Falls back to ``display_label`` when no
    short form is defined for a label.

    Case-insensitive on the override lookup so already-titled inputs like
    "Electronics And Hardware" still resolve to "Electronics" instead of
    falling through to the verbose Title Case fallback.
    """
    raw = strip_label_prefix(label).lower()
    return _SHORT_OVERRIDES.get(raw, _DISPLAY_OVERRIDES.get(raw, raw.title()))


def merge_typo_labels(
    df: pd.DataFrame,
    *,
    label_col: str = "label",
    sum_cols: tuple[str, ...] = _MARGINAL_SUM_COLS,
) -> pd.DataFrame:
    df = df.copy()
    df[label_col] = df[label_col].replace(TYPO_LABEL, CANONICAL_LABEL)
    agg = {col: "sum" for col in sum_cols if col in df.columns}
    other_cols = [c for c in df.columns if c not in agg and c != label_col]
    for col in other_cols:
        agg[col] = "first"
    merged = df.groupby(label_col, as_index=False).agg(agg)
    if "word_count_sum" in merged.columns and "word_count_docs" in merged.columns:
        merged["word_count_mean"] = np.where(
            merged["word_count_docs"] > 0,
            merged["word_count_sum"] / merged["word_count_docs"],
            np.nan,
        )
    if "score_sum" in merged.columns and "doc_count" in merged.columns:
        merged["score_mean"] = np.where(
            merged["doc_count"] > 0,
            merged["score_sum"] / merged["doc_count"],
            np.nan,
        )
    return merged


def add_token_estimate(
    df: pd.DataFrame,
    multiplier: float = WORD_TO_TOKEN_MULTIPLIER,
) -> pd.DataFrame:
    df = df.copy()
    df["token_count_est"] = (df["word_count_sum"] * multiplier).astype(int)
    return df


def resolve_eda_paths(eda_dir: Path) -> dict[str, Path]:
    eda_dir = Path(eda_dir)
    resolved = {
        name: eda_dir / filename for name, filename in _EDA_REQUIRED_FILES.items()
    }
    missing = [str(path.name) for path in resolved.values() if not path.exists()]
    if missing:
        joined = ", ".join(sorted(missing))
        raise FileNotFoundError(f"Missing required EDA files in {eda_dir}: {joined}")
    return resolved


def load_marginals(eda_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = resolve_eda_paths(eda_dir)
    topic_df = pd.read_csv(paths["topic"])
    format_df = pd.read_csv(paths["format"])
    topic_df = merge_typo_labels(topic_df, label_col="label")
    format_df = merge_typo_labels(format_df, label_col="label")
    topic_df = add_token_estimate(topic_df)
    format_df = add_token_estimate(format_df)
    return topic_df, format_df


def load_joint(eda_dir: Path) -> pd.DataFrame:
    paths = resolve_eda_paths(eda_dir)
    df = pd.read_csv(paths["joint"])
    df["topic_label"] = df["topic_label"].replace(TYPO_LABEL, CANONICAL_LABEL)
    agg = {col: "sum" for col in _JOINT_SUM_COLS if col in df.columns}
    other_cols = [
        c
        for c in df.columns
        if c not in agg and c not in ("topic_label", "format_label")
    ]
    for col in other_cols:
        agg[col] = "first"
    merged = df.groupby(["topic_label", "format_label"], as_index=False).agg(agg)
    if "word_count_sum" in merged.columns and "word_count_docs" in merged.columns:
        merged["word_count_mean"] = np.where(
            merged["word_count_docs"] > 0,
            merged["word_count_sum"] / merged["word_count_docs"],
            np.nan,
        )
    merged = add_token_estimate(merged)
    return merged


def build_grid(
    joint_df: pd.DataFrame,
    topics: list[str] | None = None,
    formats: list[str] | None = None,
) -> pd.DataFrame:
    topics = topics or list(CANONICAL_TOPIC_LABELS)
    formats = formats or list(CANONICAL_FORMAT_LABELS)
    full = pd.DataFrame(
        list(itertools.product(topics, formats)),
        columns=["topic_label", "format_label"],
    )
    merged = full.merge(joint_df, on=["topic_label", "format_label"], how="left")
    fill_cols = [
        "doc_count",
        "word_count_sum",
        "word_count_docs",
        "token_count",
        "token_count_est",
    ]
    for col in fill_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0).astype(int)
    return merged


__all__ = [
    "add_token_estimate",
    "build_grid",
    "display_label",
    "load_joint",
    "load_marginals",
    "merge_typo_labels",
    "resolve_eda_paths",
    "short_label",
    "strip_label_prefix",
]
