"""Composite correctness-differential influence 2x2 panel figure."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dolma.distribution_report.data_loader import short_label
from dolma.distribution_report.influence_composite import (
    _PANEL_ORDER,
    FIGURE_HEIGHT_GRID_PX,
    FIGURE_WIDTH_GRID_PX,
)
from dolma.distribution_report.influence_figures import _benchmark_display, _pivot_matrix
from dolma.distribution_report.influence_loader import canonical_axis_order
from dolma.distribution_report.style import (
    COLOR_EMPTY_CELL,
    DIVERGING_COLORSCALE,
    INFLUENCE_SCALE_FACTOR,
    apply_colorbar_strip_axes,
    apply_heatmap_axes,
    diverging_colorbar_strip,
    paper_layout,
    plotly_font,
    save_figure,
    symmetric_color_limit,
)


def fig_influence_diff_correct_2x2(
    correct_grids: dict[str, pd.DataFrame],
    incorrect_grids: dict[str, pd.DataFrame],
    output_dir: Path,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> None:
    topic_order, format_order = canonical_axis_order()
    x_labels = [short_label(f) for f in format_order]
    y_labels = [short_label(t) for t in topic_order]

    matrices: dict[str, np.ndarray] = {}
    for key, _, _ in _PANEL_ORDER:
        if key not in correct_grids or key not in incorrect_grids:
            continue
        m_correct = _pivot_matrix(correct_grids[key], "mean_score", topic_order, format_order) * INFLUENCE_SCALE_FACTOR
        m_incorrect = _pivot_matrix(incorrect_grids[key], "mean_score", topic_order, format_order) * INFLUENCE_SCALE_FACTOR
        matrices[key] = m_correct - m_incorrect

    if not matrices:
        return

    # One shared scale across all four panels keeps the correctness
    # differential comparable across benchmarks.
    all_finite = np.concatenate(
        [m[np.isfinite(m)].ravel() for m in matrices.values()]
    )
    abs_max = symmetric_color_limit(all_finite)
    colorbar_values, colorbar_ticks, colorbar_text = diverging_colorbar_strip(
        abs_max,
        n_ticks=7,
        decimals=1,
    )

    subplot_title_by_cell = {
        (row, col): f"<b>{_benchmark_display(key)}</b>"
        for key, row, col in _PANEL_ORDER
        if key in matrices
    }
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[
            subplot_title_by_cell.get((1, 1), ""),
            subplot_title_by_cell.get((1, 2), ""),
            "",
            subplot_title_by_cell.get((2, 1), ""),
            subplot_title_by_cell.get((2, 2), ""),
        ],
        specs=[
            [{}, {}, {"rowspan": 2}],
            [{}, {}, None],
        ],
        column_widths=[0.47, 0.47, 0.035],
        horizontal_spacing=0.06, vertical_spacing=0.03,
    )
    for key, row, col in _PANEL_ORDER:
        if key not in matrices:
            continue
        fig.add_trace(
            go.Heatmap(
                z=matrices[key], x=x_labels, y=y_labels,
                colorscale=DIVERGING_COLORSCALE,
                zmin=-abs_max,
                zmax=abs_max,
                zmid=0,
                showscale=False,
                xgap=1, ygap=1,
            ),
            row=row, col=col,
        )
    fig.add_trace(
        go.Heatmap(
            z=colorbar_values,
            x=[""],
            y=np.linspace(-abs_max, abs_max, colorbar_values.shape[0]),
            colorscale=DIVERGING_COLORSCALE,
            zmin=-abs_max,
            zmax=abs_max,
            zmid=0,
            showscale=False,
            xgap=0,
            ygap=0,
        ),
        row=1,
        col=3,
    )

    # No in-image title \u2014 LaTeX caption covers it.
    base_layout = paper_layout(title=None, plot_bgcolor=COLOR_EMPTY_CELL)
    base_layout["margin"]["r"] = 210
    fig.update_layout(
        **base_layout,
        width=FIGURE_WIDTH_GRID_PX, height=FIGURE_HEIGHT_GRID_PX,
    )
    fig.update_layout(margin={"l": 200, "r": 210, "t": 30, "b": 160})
    fig.update_annotations(font=plotly_font("SUBPLOT_TITLE"))
    apply_heatmap_axes(fig, x_labels=x_labels, y_labels=y_labels)
    fig.update_xaxes(showticklabels=False, row=1)
    fig.update_yaxes(showticklabels=False, col=2)
    apply_colorbar_strip_axes(
        fig,
        row=1,
        col=3,
        tick_vals=colorbar_ticks,
        tick_text=colorbar_text,
        title="Correct − Incorrect (z-score)",
        title_x=1.08,
        title_y=0.5,
    )
    fig.add_annotation(
        text="Format",
        x=0.5,
        y=-0.10,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=plotly_font("AXIS_TITLE"),
    )
    fig.add_annotation(
        text="Topic",
        x=-0.11,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        textangle=-90,
        font=plotly_font("AXIS_TITLE"),
    )
    save_figure(
        fig, output_dir, "fig_influence_diff_correct_2x2",
        formats, FIGURE_WIDTH_GRID_PX, FIGURE_HEIGHT_GRID_PX,
    )


def fig_influence_diff_canonical_2panel(
    grids: dict[str, pd.DataFrame],
    output_dir: Path,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> None:
    """1x2 diff heatmap matching fig-diff-combined-canonical.tex caption.

    Left panel : SocialIQA - ARC-Challenge   (reasoning contrast)
    Right panel: MMLU Social Sciences - MMLU STEM   (knowledge contrast)

    Replaces the misuse of fig_influence_socialiqa_combined that Round 5's
    Stage-2 regen produced a 3-panel SocialIQA-vs-X layout for. Uses a
    single shared coloraxis so the colorbar shows the full diverging
    gradient (per the colorbar-tickmode lesson in influence_composite.py).
    """
    pairs = [
        ("queries_socialiqa", "queries_arc_challenge",
         "Reasoning contrast (SocialIQA − ARC-Challenge)"),
        ("queries_mmlu_social_science", "queries_mmlu_stem",
         "Knowledge contrast (MMLU Social Sci − MMLU STEM)"),
    ]
    available = [(a, b, title) for a, b, title in pairs
                  if a in grids and b in grids]
    if not available:
        return

    topic_order, format_order = canonical_axis_order()
    x_labels = [_label(f) for f in format_order]
    y_labels = [_label(t) for t in topic_order]

    diffs: list[np.ndarray] = []
    for a, b, _ in available:
        m_a = _pivot_matrix(grids[a], "mean_score", topic_order, format_order) * INFLUENCE_SCALE_FACTOR
        m_b = _pivot_matrix(grids[b], "mean_score", topic_order, format_order) * INFLUENCE_SCALE_FACTOR
        diffs.append(m_a - m_b)

    finite = np.concatenate([d[np.isfinite(d)].ravel() for d in diffs])
    if len(finite) == 0:
        return
    abs_max = symmetric_color_limit(finite)
    colorbar_values, colorbar_ticks, colorbar_text = diverging_colorbar_strip(
        abs_max,
        n_ticks=7,
        decimals=1,
    )
    figure_width = 2200
    figure_height = 1040

    subplot_titles = [f"<b>{title}</b>" for _, _, title in available]
    fig = make_subplots(
        rows=1, cols=len(available) + 1,
        subplot_titles=[*subplot_titles, ""],
        column_widths=[0.475, 0.475, 0.035],
        horizontal_spacing=0.055,
    )
    for col_idx, ((a, b, _), diff) in enumerate(zip(available, diffs), start=1):
        fig.add_trace(
            go.Heatmap(
                z=diff, x=x_labels, y=y_labels,
                colorscale=DIVERGING_COLORSCALE,
                zmin=-abs_max,
                zmax=abs_max,
                zmid=0,
                showscale=False,
                xgap=1, ygap=1,
            ),
            row=1, col=col_idx,
        )
        if col_idx > 1:
            fig.update_yaxes(showticklabels=False, row=1, col=col_idx)
    fig.add_trace(
        go.Heatmap(
            z=colorbar_values,
            x=[""],
            y=np.linspace(-abs_max, abs_max, colorbar_values.shape[0]),
            colorscale=DIVERGING_COLORSCALE,
            zmin=-abs_max,
            zmax=abs_max,
            zmid=0,
            showscale=False,
            xgap=0,
            ygap=0,
        ),
        row=1,
        col=len(available) + 1,
    )

    base_layout = paper_layout(title=None, plot_bgcolor=COLOR_EMPTY_CELL)
    fig.update_layout(
        **base_layout,
        width=figure_width,
        height=figure_height,
    )
    fig.update_layout(margin={"l": 180, "r": 210, "t": 45, "b": 135})
    fig.update_annotations(font=plotly_font("SUBPLOT_TITLE", scale=0.6))
    apply_heatmap_axes(fig, x_labels=x_labels, y_labels=y_labels, tick_scale=0.6)
    fig.update_xaxes(
        showticklabels=False,
        ticks="",
        showline=False,
        row=1,
        col=len(available) + 1,
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=colorbar_ticks,
        ticktext=colorbar_text,
        tickfont=plotly_font("COLORBAR_TICK", scale=0.8),
        ticks="outside",
        ticklen=4,
        tickwidth=1,
        tickcolor="#222222",
        side="right",
        showline=False,
        row=1,
        col=len(available) + 1,
    )
    fig.add_annotation(
        text="Format", x=0.5, y=-0.10, xref="paper", yref="paper",
        showarrow=False, font=plotly_font("AXIS_TITLE", scale=0.6),
    )
    fig.add_annotation(
        text="Topic", x=-0.07, y=0.5, xref="paper", yref="paper",
        showarrow=False, textangle=-90, font=plotly_font("AXIS_TITLE", scale=0.6),
    )
    fig.add_annotation(
        text="Signed difference (z-score)",
        x=1.08,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        textangle=-90,
        font=plotly_font("COLORBAR_TICK", scale=0.9),
    )
    save_figure(
        fig, output_dir, "fig_influence_diff_canonical_2panel",
        formats, figure_width, figure_height,
    )


# Local alias keeps the new generator consistent with the short_label
# rollout: every short_label call site in this module already uses
# `short_label`; reusing it for the axis labels keeps things tidy.
_label = short_label


__all__ = [
    "fig_influence_diff_canonical_2panel",
    "fig_influence_diff_correct_2x2",
]
