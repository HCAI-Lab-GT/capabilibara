"""Combined canonical difference figure: SocialIQA vs three non-social benchmarks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dolma.distribution_report.data_loader import short_label
from dolma.distribution_report.influence_figures import _pivot_matrix
from dolma.distribution_report.influence_loader import canonical_axis_order
from dolma.distribution_report.style import (
    COLOR_EMPTY_CELL,
    COMPARISON_COLORSCALE,
    INFLUENCE_SCALE_FACTOR,
    INFLUENCE_UNIT,
    diverging_colorbar_ticks,
    paper_layout,
    plotly_font,
    save_figure,
)

_WIDE_SCALE = 0.6  # figure* composite — compensate for less LaTeX downscaling

_SOCIALIQA_PAIRS = [
    ("queries_gsm8k", "SocialIQA \u2212 GSM8K"),
    ("queries_arc_challenge", "SocialIQA \u2212 ARC-Challenge"),
    ("queries_arc_easy", "SocialIQA \u2212 ARC-Easy"),
]

FIGURE_WIDTH_SOCIALIQA_3WAY_PX = 3600
FIGURE_HEIGHT_SOCIALIQA_3WAY_PX = 900


def fig_influence_socialiqa_combined(
    grids: dict[str, pd.DataFrame],
    output_dir: Path,
    formats: tuple[str, ...] = ("pdf", "png"),
    suffix: str = "",
) -> None:
    socialiqa_key = "queries_socialiqa"
    if socialiqa_key not in grids:
        return
    available = [(key, label) for key, label in _SOCIALIQA_PAIRS if key in grids]
    if not available:
        return

    topic_order, format_order = canonical_axis_order()
    x_labels = [short_label(f) for f in format_order]
    y_labels = [short_label(t) for t in topic_order]

    base = _pivot_matrix(grids[socialiqa_key], "mean_score", topic_order, format_order) * INFLUENCE_SCALE_FACTOR
    diffs = [
        base - _pivot_matrix(grids[key], "mean_score", topic_order, format_order) * INFLUENCE_SCALE_FACTOR
        for key, _ in available
    ]

    finite = np.concatenate([d[np.isfinite(d)].ravel() for d in diffs])
    if len(finite) == 0:
        return
    abs_max = float(max(np.percentile(np.abs(finite), 95), 1e-8))
    tick_vals, tick_text = diverging_colorbar_ticks(abs_max, n_ticks=5, decimals=2)

    ncols = len(available)
    fig = make_subplots(
        rows=1, cols=ncols,
        subplot_titles=[f"<b>{label}</b>" for _, label in available],
        horizontal_spacing=0.03,
    )
    for col_idx, ((_, label), diff) in enumerate(zip(available, diffs), start=1):
        show_colorbar = col_idx == ncols
        fig.add_trace(
            go.Heatmap(
                z=diff, x=x_labels, y=y_labels,
                coloraxis="coloraxis",
                xgap=1, ygap=1,
                name=label,
            ),
            row=1, col=col_idx,
        )
        if not show_colorbar:
            fig.update_yaxes(showticklabels=False, row=1, col=col_idx + 1)

    fig.update_layout(
        **paper_layout(
            f"Signed Influence Difference (difference of {INFLUENCE_UNIT}s): SocialIQA vs non-social benchmarks",
            plot_bgcolor=COLOR_EMPTY_CELL,
        ),
        coloraxis={
            "colorscale": COMPARISON_COLORSCALE,
            "cmid": 0, "cmin": -abs_max, "cmax": abs_max,
            "colorbar": {
                "title": "", "thickness": 12,
                "tickmode": "array", "tickvals": tick_vals, "ticktext": tick_text,
            },
        },
        width=FIGURE_WIDTH_SOCIALIQA_3WAY_PX,
        height=FIGURE_HEIGHT_SOCIALIQA_3WAY_PX,
    )
    fig.update_xaxes(tickangle=45, tickfont=plotly_font("TICK", scale=_WIDE_SCALE))
    fig.update_yaxes(tickfont=plotly_font("TICK", scale=_WIDE_SCALE))
    fig.update_annotations(font=plotly_font("SUBPLOT_TITLE", scale=_WIDE_SCALE))
    fig.layout.margin.l = 200
    fig.layout.margin.b = 100
    fig.add_annotation(
        text="Format", x=0.5, y=-0.10, xref="paper", yref="paper",
        showarrow=False, font=plotly_font("AXIS_TITLE", scale=_WIDE_SCALE),
    )
    fig.add_annotation(
        text="Topic", x=-0.06, y=0.5, xref="paper", yref="paper",
        showarrow=False, textangle=-90, font=plotly_font("AXIS_TITLE", scale=_WIDE_SCALE),
    )
    save_figure(
        fig, output_dir,
        f"fig_influence_diff_socialiqa_combined{suffix}",
        formats,
        FIGURE_WIDTH_SOCIALIQA_3WAY_PX,
        FIGURE_HEIGHT_SOCIALIQA_3WAY_PX,
    )


__all__ = [
    "FIGURE_HEIGHT_SOCIALIQA_3WAY_PX",
    "FIGURE_WIDTH_SOCIALIQA_3WAY_PX",
    "fig_influence_socialiqa_combined",
]
