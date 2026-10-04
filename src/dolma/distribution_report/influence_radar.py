"""Radar chart showing topic-level influence fingerprint per benchmark."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from dolma.distribution_report.data_loader import short_label
from dolma.distribution_report.influence_figures import _benchmark_display
from dolma.distribution_report.style import (
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    FIGURE_HEIGHT_HEATMAP_PX,
    FIGURE_WIDTH_HEATMAP_PX,
    paper_layout,
    save_figure,
)

_RADAR_COLORS = [
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    "#009E73",
    "#E69F00",
]


def _topic_marginals(
    grid_df: pd.DataFrame,
    value_col: str = "mean_score",
) -> pd.DataFrame:
    return (
        grid_df.groupby("topic_label", as_index=False)
        .agg(value=(value_col, "mean"))
        .sort_values("topic_label")
    )


def fig_influence_radar(
    grids: dict[str, pd.DataFrame],
    output_dir: Path,
    value_col: str = "abs_mean_score",
    formats: tuple[str, ...] = ("pdf", "png"),
) -> None:
    fig = go.Figure()

    topic_order = None
    for i, (key, grid) in enumerate(grids.items()):
        mdf = _topic_marginals(grid, value_col)
        if topic_order is None:
            topic_order = mdf["topic_label"].tolist()
        labels = [short_label(t) for t in mdf["topic_label"]]
        values = mdf["value"].tolist()
        values_closed = values + [values[0]]
        labels_closed = labels + [labels[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=values_closed,
                theta=labels_closed,
                name=_benchmark_display(key),
                line={"color": _RADAR_COLORS[i % len(_RADAR_COLORS)]},
                fill="toself",
                opacity=0.3,
            )
        )

    suffix = "signed" if value_col == "mean_score" else "abs"
    cbar_label = "Mean Influence" if value_col == "mean_score" else "Mean |Influence|"
    fig.update_layout(
        **paper_layout(f"Topic Influence Fingerprint ({cbar_label})"),
        polar={
            "radialaxis": {"visible": True, "showticklabels": True},
        },
        width=FIGURE_WIDTH_HEATMAP_PX,
        height=FIGURE_HEIGHT_HEATMAP_PX,
        showlegend=True,
    )
    save_figure(
        fig,
        output_dir,
        f"fig_influence_radar_{suffix}",
        formats,
        FIGURE_WIDTH_HEATMAP_PX,
        FIGURE_HEIGHT_HEATMAP_PX,
    )


__all__ = [
    "fig_influence_radar",
]
