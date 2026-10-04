"""Composite absolute topic influence bar figure."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dolma.distribution_report.data_loader import short_label
from dolma.distribution_report.influence_composite import _PANEL_ORDER
from dolma.distribution_report.influence_figures import _benchmark_display
from dolma.distribution_report.influence_loader import canonical_axis_order
from dolma.distribution_report.style import (
    COLOR_PRIMARY,
    INFLUENCE_SCALE_FACTOR,
    INFLUENCE_UNIT,
    paper_layout,
    plotly_font,
    save_figure,
)

FIGURE_WIDTH_4BAR_PX = 1400
FIGURE_HEIGHT_4BAR_PX = 1400


def fig_influence_abs_topic_4panel(
    grids: dict[str, pd.DataFrame],
    output_dir: Path,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> None:
    topic_order, _ = canonical_axis_order()

    topic_abs: dict[str, pd.Series] = {}
    for key, _, _ in _PANEL_ORDER:
        if key not in grids:
            continue
        agg = (
            grids[key]
            .groupby("topic_label")["abs_mean_score"]
            .mean()
            .reindex(topic_order)
            .fillna(0)
            .mul(INFLUENCE_SCALE_FACTOR)
        )
        topic_abs[key] = agg

    if not topic_abs:
        return

    anchor_key = "queries_socialiqa"
    if anchor_key not in topic_abs:
        anchor_key = next(iter(topic_abs))
    sorted_topics = topic_abs[anchor_key].sort_values(ascending=True).index.tolist()
    y_labels = [short_label(t) for t in sorted_topics]

    all_vals = np.concatenate([s.values for s in topic_abs.values()])
    x_max = float(np.nanmax(all_vals)) * 1.05

    subplot_titles = [f"<b>{_benchmark_display(key)}</b>" for key, _, _ in _PANEL_ORDER if key in topic_abs]
    fig = make_subplots(
        rows=2, cols=2, subplot_titles=subplot_titles,
        horizontal_spacing=0.06, vertical_spacing=0.05, shared_xaxes=False,
    )
    for key, row, col in _PANEL_ORDER:
        if key not in topic_abs:
            continue
        values = [topic_abs[key].get(t, 0.0) for t in sorted_topics]
        fig.add_trace(
            go.Bar(x=values, y=y_labels, orientation="h", marker_color=COLOR_PRIMARY, showlegend=False),
            row=row, col=col,
        )
        fig.update_xaxes(range=[0, x_max], row=row, col=col)
        if col == 2:
            fig.update_yaxes(showticklabels=False, row=row, col=col)

    fig.update_layout(
        **paper_layout(title=None),
        width=FIGURE_WIDTH_4BAR_PX, height=FIGURE_HEIGHT_4BAR_PX,
    )
    fig.layout.margin.t = 30
    fig.update_annotations(font=plotly_font("SUBPLOT_TITLE"))
    fig.update_xaxes(tickfont=plotly_font("TICK"))
    fig.update_yaxes(tickfont=plotly_font("TICK"))
    fig.update_xaxes(title_text=INFLUENCE_UNIT,
                     title_font=plotly_font("AXIS_TITLE"), row=2)
    save_figure(
        fig, output_dir, "fig_influence_abs_topic_4panel",
        formats, FIGURE_WIDTH_4BAR_PX, FIGURE_HEIGHT_4BAR_PX,
    )


__all__ = [
    "FIGURE_HEIGHT_4BAR_PX",
    "FIGURE_WIDTH_4BAR_PX",
    "fig_influence_abs_topic_4panel",
]
