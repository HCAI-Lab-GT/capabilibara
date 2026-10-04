"""Concentration and distribution metrics for taxonomy analysis."""

from __future__ import annotations

import numpy as np


def gini_coefficient(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    index = np.arange(1, n + 1)
    return float(
        (2.0 * np.sum(index * sorted_vals) / (n * np.sum(sorted_vals))) - (n + 1) / n
    )


def effective_bins(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[values > 0]
    if len(values) == 0:
        return 0.0
    proportions = values / values.sum()
    entropy = -np.sum(proportions * np.log(proportions))
    return float(np.exp(entropy))


def concentration_stats(grid_df) -> dict:
    doc_counts = grid_df["doc_count"].values.astype(float)
    token_counts = grid_df["token_count_est"].values.astype(float)

    total_tokens = token_counts.sum()
    sorted_tokens = np.sort(token_counts)[::-1]
    cumulative = np.cumsum(sorted_tokens)

    top_10_share = (
        float(cumulative[9] / total_tokens) if len(sorted_tokens) >= 10 else 1.0
    )
    top_50_share = (
        float(cumulative[49] / total_tokens) if len(sorted_tokens) >= 50 else 1.0
    )
    top_100_share = (
        float(cumulative[99] / total_tokens) if len(sorted_tokens) >= 100 else 1.0
    )

    return {
        "total_bins": len(doc_counts),
        "empty_bins": int(np.sum(doc_counts == 0)),
        "near_empty_100": int(np.sum((doc_counts > 0) & (doc_counts < 100))),
        "near_empty_1000": int(np.sum((doc_counts > 0) & (doc_counts < 1000))),
        "gini_doc": gini_coefficient(doc_counts),
        "gini_token": gini_coefficient(token_counts),
        "effective_bins_doc": effective_bins(doc_counts),
        "effective_bins_token": effective_bins(token_counts),
        "top_10_token_share": top_10_share,
        "top_50_token_share": top_50_share,
        "top_100_token_share": top_100_share,
    }


__all__ = [
    "concentration_stats",
    "effective_bins",
    "gini_coefficient",
]
