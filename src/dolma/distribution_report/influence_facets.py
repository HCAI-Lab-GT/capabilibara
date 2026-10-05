"""Format-conditioned topic influence bar charts (faceted)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dolma.distribution_report.data_loader import short_label
from dolma.distribution_report.influence_figures import _benchmark_display
from dolma.distribution_report.style import (
    COLOR_PRIMARY,
    FIGURE_HEIGHT_SIDE_BY_SIDE_PX,
    FIGURE_WIDTH_SIDE_BY_SIDE_PX,
    paper_layout,
    save_figure,
)

KEY_FORMATS = [
    "__label__q_a_forum",
    "__label__academic_writing",
    "__label__knowledge_article",
    "__label__tutorial",
]


def fig_influence_format_facets(
    grid_df: pd.DataFrame,
    benchmark_key: str,
    output_dir: Path,
    format_keys: list[str] | None = None,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> None:
    format_keys = format_keys or KEY_FORMATS
    available = set(grid_df["format_label"].unique())
    format_keys = [f for f in format_keys if f in available]
    if not format_keys:
        return

    n_cols = len(format_keys)
    name = _benchmark_display(benchmark_key)
    short = benchmark_key.removeprefix("queries_")

    fig = make_subplots(
        rows=1,
        cols=n_cols,
        subplot_titles=[short_label(f) for f in format_keys],
        horizontal_spacing=0.08,
    )

    for col_idx, fmt_key in enumerate(format_keys, start=1):
        subset = grid_df[grid_df["format_label"] == fmt_key].copy()
        subset = subset.sort_values("mean_score", ascending=True)
        labels = [short_label(t) for t in subset["topic_label"]]
        values = subset["mean_score"].tolist()

        fig.add_trace(
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker_color=COLOR_PRIMARY,
                showlegend=False,
            ),
            row=1,
            col=col_idx,
        )

    fig.update_layout(
        **paper_layout(f"{name}: Topic Influence by Format"),
        width=FIGURE_WIDTH_SIDE_BY_SIDE_PX,
        height=FIGURE_HEIGHT_SIDE_BY_SIDE_PX,
    )
    save_figure(
        fig,
        output_dir,
        f"fig_influence_facets_{short}",
        formats,
        FIGURE_WIDTH_SIDE_BY_SIDE_PX,
        FIGURE_HEIGHT_SIDE_BY_SIDE_PX,
    )


__all__ = [
    "fig_influence_format_facets",
]
