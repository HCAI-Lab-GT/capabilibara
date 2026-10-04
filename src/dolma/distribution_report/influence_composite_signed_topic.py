"""Composite signed topic and format influence bar figures (4-panel, 2x2) — Plotly with domain/task color+pattern encoding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dolma.distribution_report.data_loader import short_label
from dolma.distribution_report.influence_composite import _PANEL_ORDER
from dolma.distribution_report.influence_figures import _benchmark_display
from dolma.distribution_report.style import (
    INFLUENCE_SCALE_FACTOR,
    INFLUENCE_UNIT,
    paper_layout,
    plotly_font,
    save_figure,
)

# Wide (figure*) composite — LaTeX downscales less, so shrink Plotly fonts to match.
_WIDE_SCALE = 0.6

FIGURE_WIDTH_4BAR_PX = 2200
FIGURE_HEIGHT_4BAR_PX = 1400

# Color = domain (social vs math), pattern = task type (reasoning vs knowledge)
_PANEL_STYLE: dict[str, dict] = {
    "queries_socialiqa": {
        "color": "#984EA3",
        "pattern_shape": "",       # solid — reasoning
        "label": "SocialIQA (reasoning)",
    },
    "queries_mmlu_social_science": {
        "color": "#984EA3",
        "pattern_shape": "/",      # hatched — knowledge
        "label": "MMLU Social Sci (knowledge)",
    },
    "queries_arc_challenge": {
        "color": "#1b7837",
        "pattern_shape": "",       # solid — reasoning
        "label": "ARC-Challenge (reasoning)",
    },
    "queries_mmlu_stem": {
        "color": "#1b7837",
        "pattern_shape": "/",      # hatched — knowledge
        "label": "MMLU STEM (knowledge)",
    },
}


def _signed_marginals(
    grids: dict[str, pd.DataFrame],
    panel_order: list[tuple[str, int, int]],
    group_col: str,
) -> dict[str, pd.Series]:
    result: dict[str, pd.Series] = {}
    for key, _, _ in panel_order:
        if key not in grids:
            continue
        result[key] = (
            grids[key]
            .groupby(group_col)["mean_score"]
            .mean()
            .mul(INFLUENCE_SCALE_FACTOR)
        )
    return result


def _signed_panel_figure(
    marginals: dict[str, pd.Series],
    panel_order: list[tuple[str, int, int]],
    axis_label: str,
    title: str,
    filename: str,
    output_dir: Path,
    formats: tuple[str, ...],
    nrows: int = 2,
    ncols: int = 2,
    width: int = FIGURE_WIDTH_4BAR_PX,
    height: int = FIGURE_HEIGHT_4BAR_PX,
) -> None:
    if not marginals:
        return

    all_vals = np.concatenate([s.values for s in marginals.values()])
    finite = all_vals[np.isfinite(all_vals)]
    if not len(finite):
        return

    subplot_titles = [
        f"<b>{_benchmark_display(key)}</b>"
        for key, _, _ in panel_order
        if key in marginals
    ]
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.05,
        shared_xaxes=False,
        shared_yaxes=False,
    )

    # Track which legend entries have already been added (deduplicate across panels)
    _legend_shown: set[str] = set()

    for key, row, col in panel_order:
        if key not in marginals:
            continue

        sorted_vals = marginals[key].sort_values(ascending=True)
        y_labels = [short_label(t) for t in sorted_vals.index]
        values = sorted_vals.tolist()

        panel_min = float(np.nanmin(sorted_vals.values))
        panel_max = float(np.nanmax(sorted_vals.values))
        margin = 0.02 * (panel_max - panel_min) if panel_max > panel_min else 0.01

        style = _PANEL_STYLE.get(key, {})
        color = style.get("color", "#4477AA")
        pattern_shape = style.get("pattern_shape", "")
        label = style.get("label", key)
        show_legend = label not in _legend_shown
        if show_legend:
            _legend_shown.add(label)

        fig.add_trace(
            go.Bar(
                x=values,
                y=y_labels,
                orientation="h",
                name=label,
                legendgroup=label,
                showlegend=show_legend,
                marker={
                    "color": color,
                    "pattern": {
                        "shape": pattern_shape,
                        "solidity": 0.4,          # hatch density — lower = more open
                        "fgcolor": "white",        # hatch line color against the base color
                        "size": 8,
                    },
                    "line": {"color": color, "width": 0.5},
                },
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(range=[panel_min - margin, panel_max + margin], row=row, col=col)

    fig.update_layout(
        **paper_layout(title=None),
        width=width,
        height=height,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "center",
            "x": 0.5,
            "font": plotly_font("LEGEND", scale=_WIDE_SCALE),
            "tracegroupgap": 0,
        },
    )
    fig.layout.margin.t = 80   # legend space; no in-image title
    fig.layout.margin.l = 200
    fig.update_annotations(font=plotly_font("SUBPLOT_TITLE", scale=_WIDE_SCALE))
    fig.update_xaxes(tickfont=plotly_font("TICK", scale=_WIDE_SCALE))
    fig.update_yaxes(tickfont=plotly_font("TICK", scale=_WIDE_SCALE))
    # Only label x-axis on bottom row panels
    for c in range(1, ncols + 1):
        fig.update_xaxes(
            title_text=INFLUENCE_UNIT,
            title_font=plotly_font("AXIS_TITLE", scale=_WIDE_SCALE),
            row=nrows, col=c,
        )
    fig.add_annotation(
        text=axis_label,
        x=-0.10,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        textangle=-90,
        font=plotly_font("AXIS_TITLE", scale=_WIDE_SCALE),
    )
    save_figure(fig, output_dir, filename, formats, width, height)


# Round 8: 4x1 vertical panel order (user feedback: "we could separate
# it and have them stacked all four on top of each other like 1 2 3 4").
# Reuses _PANEL_ORDER's benchmark sequence; remaps (row, col) to a single
# column.
_PANEL_ORDER_4X1 = [
    (key, idx + 1, 1) for idx, (key, _, _) in enumerate(_PANEL_ORDER)
]


def fig_influence_signed_topic_4panel(
    grids: dict[str, pd.DataFrame],
    output_dir: Path,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> None:
    marginals = _signed_marginals(grids, _PANEL_ORDER_4X1, "topic_label")
    _signed_panel_figure(
        marginals,
        _PANEL_ORDER_4X1,
        axis_label="Topic",
        title=f"Mean Signed Influence by Topic ({INFLUENCE_UNIT})",
        filename="fig_influence_signed_topic_4panel",
        output_dir=output_dir,
        formats=formats,
        nrows=4, ncols=1,
        # Tall and narrow to suit a 4x1 stack on a single-column COLM page.
        width=FIGURE_WIDTH_4BAR_PX // 2,
        height=int(FIGURE_HEIGHT_4BAR_PX * 1.4),
    )


def fig_influence_signed_format_4panel(
    grids: dict[str, pd.DataFrame],
    output_dir: Path,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> None:
    marginals = _signed_marginals(grids, _PANEL_ORDER_4X1, "format_label")
    _signed_panel_figure(
        marginals,
        _PANEL_ORDER_4X1,
        axis_label="Format",
        title=f"Mean Signed Influence by Format ({INFLUENCE_UNIT})",
        filename="fig_influence_signed_format_4panel",
        output_dir=output_dir,
        formats=formats,
        nrows=4, ncols=1,
        width=FIGURE_WIDTH_4BAR_PX // 2,
        height=int(FIGURE_HEIGHT_4BAR_PX * 1.4),
    )


__all__ = [
    "FIGURE_HEIGHT_4BAR_PX",
    "FIGURE_WIDTH_4BAR_PX",
    "fig_influence_signed_format_4panel",
    "fig_influence_signed_topic_4panel",
]