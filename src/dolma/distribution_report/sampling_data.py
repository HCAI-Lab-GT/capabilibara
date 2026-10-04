"""Shared sampling-comparison data helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _pivot_grid(
    grid_df: pd.DataFrame,
    value_col: str,
    topic_order: list[str] | None = None,
    format_order: list[str] | None = None,
) -> pd.DataFrame:
    matrix = (
        grid_df.pivot(index="topic_label", columns="format_label", values=value_col)
        .fillna(0)
        .astype(float)
    )
    if topic_order is None:
        topic_order = matrix.sum(axis=1).sort_values(ascending=False).index.tolist()
    if format_order is None:
        format_order = matrix.sum(axis=0).sort_values(ascending=False).index.tolist()
    return matrix.reindex(index=topic_order, columns=format_order, fill_value=0)


def comparison_axis_order(
    representative_grid: pd.DataFrame,
    value_col: str = "token_count",
) -> tuple[list[str], list[str]]:
    matrix = _pivot_grid(representative_grid, value_col)
    return matrix.index.tolist(), matrix.columns.tolist()


def comparison_matrices(
    stratified_grid: pd.DataFrame,
    representative_grid: pd.DataFrame,
    value_col: str = "token_count",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    topic_order, format_order = comparison_axis_order(representative_grid, value_col)
    strat_matrix = _pivot_grid(stratified_grid, value_col, topic_order, format_order)
    rep_matrix = _pivot_grid(representative_grid, value_col, topic_order, format_order)
    return strat_matrix, rep_matrix, topic_order, format_order


def build_share_difference_table(
    stratified_grid: pd.DataFrame,
    representative_grid: pd.DataFrame,
) -> pd.DataFrame:
    strat_matrix, rep_matrix, topic_order, format_order = comparison_matrices(
        stratified_grid,
        representative_grid,
    )
    strat_total = strat_matrix.values.sum()
    rep_total = rep_matrix.values.sum()
    if strat_total == 0 or rep_total == 0:
        raise ValueError("Sampling comparison requires positive token totals.")
    strat_share = strat_matrix / strat_total
    rep_share = rep_matrix / rep_total
    # No NaN masking: both-zero cells compute log2(epsilon/epsilon) = 0 and
    # render at the diverging-scale midpoint (white) instead of background grey.
    # Per-side zeros remain at large negative/positive ratios as before.
    epsilon = 1e-10
    log_ratio = np.log2((strat_share + epsilon) / (rep_share + epsilon)).to_numpy()
    return pd.DataFrame(
        {
            "topic_label": np.repeat(topic_order, len(format_order)),
            "format_label": np.tile(format_order, len(topic_order)),
            "stratified_share": strat_share.values.ravel(),
            "representative_share": rep_share.values.ravel(),
            "log2_share_ratio": log_ratio.ravel(),
        }
    )


__all__ = [
    "build_share_difference_table",
    "comparison_axis_order",
    "comparison_matrices",
]
