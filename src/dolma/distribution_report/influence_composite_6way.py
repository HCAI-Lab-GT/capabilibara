"""Composite signed-influence 3x2 panel figures for six-benchmark layout."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dolma.distribution_report.data_loader import short_label
from dolma.distribution_report.influence_figures import _benchmark_display, _pivot_matrix
from dolma.distribution_report.influence_loader import canonical_axis_order
from dolma.distribution_report.style import (
    COLOR_EMPTY_CELL,
    DIVERGING_COLORSCALE,
    INFLUENCE_SCALE_FACTOR,
    INFLUENCE_UNIT,
    diverging_colorbar_ticks,
    paper_layout,
    plotly_font,
    save_figure,
)

_WIDE_SCALE = 0.6  # figure* composite — compensate for less LaTeX downscaling

_PANEL_ORDER_6WAY = [
    ("queries_socialiqa", 1, 1),
    ("queries_gsm8k", 1, 2),
    ("queries_arc_challenge", 1, 3),
    ("queries_mmlu_social_science", 2, 1),
    ("queries_mmlu_stem", 2, 2),
    ("queries_arc_easy", 2, 3),
]

_PANEL_COLORAXIS_IDX_6WAY = {
    (1, 1): 1, (1, 2): 2, (1, 3): 3,
    (2, 1): 4, (2, 2): 5, (2, 3): 6,
}

# Colorbar x positions for 3-col layout (right edge of each column + small gap).
# With horizontal_spacing=0.02 and 3 cols, approx col right edges: 0.32, 0.65, 0.98.
_PANEL_COLORBAR_POS_6WAY = {
    1: {"x": 0.31, "y": 0.775, "len": 0.43, "thickness": 10},
    2: {"x": 0.64, "y": 0.775, "len": 0.43, "thickness": 10},
    3: {"x": 1.01, "y": 0.775, "len": 0.43, "thickness": 10},
    4: {"x": 0.31, "y": 0.225, "len": 0.43, "thickness": 10},
    5: {"x": 0.64, "y": 0.225, "len": 0.43, "thickness": 10},
    6: {"x": 1.01, "y": 0.225, "len": 0.43, "thickness": 10},
}

FIGURE_WIDTH_6WAY_PX = 3000
FIGURE_HEIGHT_6WAY_PX = 1600


def _build_matrices(
    grids: dict[str, pd.DataFrame],
    topic_order: list[str],
    format_order: list[str],
) -> dict[str, np.ndarray]:
    return {
        key: _pivot_matrix(grids[key], "mean_score", topic_order, format_order) * INFLUENCE_SCALE_FACTOR
        for key, _, _ in _PANEL_ORDER_6WAY
        if key in grids
    }


def fig_influence_signed_3x2(
    grids: dict[str, pd.DataFrame],
    output_dir: Path,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> None:
    topic_order, format_order = canonical_axis_order()
    x_labels = [short_label(f) for f in format_order]
    y_labels = [short_label(t) for t in topic_order]
    matrices = _build_matrices(grids, topic_order, format_order)
    if not matrices:
        return

    all_finite = np.concatenate([m[np.isfinite(m)].ravel() for m in matrices.values()])
    abs_max = float(max(np.percentile(np.abs(all_finite), 95), 1e-8))
    tick_vals, tick_text = diverging_colorbar_ticks(abs_max, n_ticks=5, decimals=2)

    subplot_titles = [
        f"<b>{_benchmark_display(key)}</b>"
        for key, _, _ in _PANEL_ORDER_6WAY
        if key in matrices
    ]
    fig = make_subplots(
        rows=2, cols=3, subplot_titles=subplot_titles,
        horizontal_spacing=0.02, vertical_spacing=0.03,
    )
    for key, row, col in _PANEL_ORDER_6WAY:
        if key not in matrices:
            continue
        show_colorbar = row == 1 and col == 3
        fig.add_trace(
            go.Heatmap(
                z=matrices[key], x=x_labels, y=y_labels,
                colorscale=DIVERGING_COLORSCALE, zmid=0, zmin=-abs_max, zmax=abs_max,
                xgap=1, ygap=1, showscale=show_colorbar,
                colorbar={
                    "title": "", "x": 1.01, "len": 0.9, "y": 0.5, "thickness": 12,
                    "tickmode": "array", "tickvals": tick_vals, "ticktext": tick_text,
                },
            ),
            row=row, col=col,
        )

    fig.update_layout(
        **paper_layout(
            f"Mean Signed Influence by Topic and Format ({INFLUENCE_UNIT})",
            plot_bgcolor=COLOR_EMPTY_CELL,
        ),
        width=FIGURE_WIDTH_6WAY_PX, height=FIGURE_HEIGHT_6WAY_PX,
    )
    fig.layout.margin.t = 80
    fig.layout.margin.l = 200
    fig.layout.margin.b = 150
    fig.update_annotations(font=plotly_font("SUBPLOT_TITLE", scale=_WIDE_SCALE))
    fig.update_xaxes(tickangle=45, tickfont=plotly_font("TICK", scale=_WIDE_SCALE), showticklabels=False, row=1)
    fig.update_xaxes(tickangle=45, tickfont=plotly_font("TICK", scale=_WIDE_SCALE), row=2)
    fig.update_yaxes(tickfont=plotly_font("TICK", scale=_WIDE_SCALE))
    fig.update_yaxes(showticklabels=False, col=2)
    fig.update_yaxes(showticklabels=False, col=3)
    fig.add_annotation(
        text="Format", x=0.5, y=-0.06, xref="paper", yref="paper",
        showarrow=False, font=plotly_font("AXIS_TITLE", scale=_WIDE_SCALE),
    )
    fig.add_annotation(
        text="Topic", x=-0.07, y=0.5, xref="paper", yref="paper",
        showarrow=False, textangle=-90, font=plotly_font("AXIS_TITLE", scale=_WIDE_SCALE),
    )
    save_figure(fig, output_dir, "fig_influence_signed_3x2", formats, FIGURE_WIDTH_6WAY_PX, FIGURE_HEIGHT_6WAY_PX)


def fig_influence_signed_3x2_perpanel(
    grids: dict[str, pd.DataFrame],
    output_dir: Path,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> None:
    topic_order, format_order = canonical_axis_order()
    x_labels = [short_label(f) for f in format_order]
    y_labels = [short_label(t) for t in topic_order]
    matrices = _build_matrices(grids, topic_order, format_order)
    if not matrices:
        return

    panel_maxes = {
        key: float(max(np.percentile(np.abs(matrices[key][np.isfinite(matrices[key])]), 95), 1e-8))
        for key in matrices
    }
    subplot_titles = [
        f"<b>{_benchmark_display(key)}</b>"
        for key, _, _ in _PANEL_ORDER_6WAY
        if key in matrices
    ]
    fig = make_subplots(
        rows=2, cols=3, subplot_titles=subplot_titles,
        horizontal_spacing=0.02, vertical_spacing=0.03,
    )
    for key, row, col in _PANEL_ORDER_6WAY:
        if key not in matrices:
            continue
        panel_idx = _PANEL_COLORAXIS_IDX_6WAY[(row, col)]
        fig.add_trace(
            go.Heatmap(
                z=matrices[key], x=x_labels, y=y_labels,
                coloraxis=f"coloraxis{panel_idx}", xgap=1, ygap=1,
            ),
            row=row, col=col,
        )

    coloraxis_kwargs: dict = {}
    for key, row, col in _PANEL_ORDER_6WAY:
        if key not in matrices:
            continue
        panel_idx = _PANEL_COLORAXIS_IDX_6WAY[(row, col)]
        panel_max = panel_maxes[key]
        cb_pos = _PANEL_COLORBAR_POS_6WAY[panel_idx]
        tick_vals, tick_text = diverging_colorbar_ticks(panel_max, n_ticks=5, decimals=2)
        coloraxis_kwargs[f"coloraxis{panel_idx}"] = {
            "colorscale": DIVERGING_COLORSCALE, "cmid": 0, "cmin": -panel_max, "cmax": panel_max,
            "colorbar": dict(
                title="", tickfont=plotly_font("COLORBAR_TICK", scale=_WIDE_SCALE),
                tickmode="array", tickvals=tick_vals, ticktext=tick_text,
                **cb_pos,
            ),
        }

    base_layout = paper_layout(
        f"Mean Signed Influence by Topic and Format, per-panel scale ({INFLUENCE_UNIT})",
        plot_bgcolor=COLOR_EMPTY_CELL,
    )
    base_layout["margin"]["r"] = 120
    fig.update_layout(
        **base_layout,
        width=FIGURE_WIDTH_6WAY_PX, height=FIGURE_HEIGHT_6WAY_PX,
        **coloraxis_kwargs,
    )
    fig.layout.margin.t = 80
    fig.layout.margin.l = 200
    fig.layout.margin.b = 150
    fig.update_annotations(font=plotly_font("SUBPLOT_TITLE", scale=_WIDE_SCALE))
    fig.update_xaxes(tickangle=45, tickfont=plotly_font("TICK", scale=_WIDE_SCALE), showticklabels=False, row=1)
    fig.update_xaxes(tickangle=45, tickfont=plotly_font("TICK", scale=_WIDE_SCALE), row=2)
    fig.update_yaxes(tickfont=plotly_font("TICK", scale=_WIDE_SCALE))
    fig.update_yaxes(showticklabels=False, col=2)
    fig.update_yaxes(showticklabels=False, col=3)
    fig.add_annotation(
        text="Format", x=0.5, y=-0.06, xref="paper", yref="paper",
        showarrow=False, font=plotly_font("AXIS_TITLE", scale=_WIDE_SCALE),
    )
    fig.add_annotation(
        text="Topic", x=-0.07, y=0.5, xref="paper", yref="paper",
        showarrow=False, textangle=-90, font=plotly_font("AXIS_TITLE", scale=_WIDE_SCALE),
    )
    save_figure(
        fig, output_dir, "fig_influence_signed_3x2_perpanel",
        formats, FIGURE_WIDTH_6WAY_PX, FIGURE_HEIGHT_6WAY_PX,
    )


__all__ = [
    "FIGURE_HEIGHT_6WAY_PX",
    "FIGURE_WIDTH_6WAY_PX",
    "_PANEL_ORDER_6WAY",
    "fig_influence_signed_3x2",
    "fig_influence_signed_3x2_perpanel",
]
