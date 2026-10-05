"""Composite signed-influence 2x2 panel figures for paper layout."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dolma.distribution_report.data_loader import short_label
from dolma.distribution_report.influence_figures import _benchmark_display, _pivot_matrix
from dolma.distribution_report.style import (
    COLOR_EMPTY_CELL,
    DIVERGING_COLORSCALE,
    INFLUENCE_SCALE_FACTOR,
    apply_heatmap_axes,
    diverging_colorbar_strip,
    paper_layout,
    plotly_font,
    save_figure,
    symmetric_color_limit,
)

_PANEL_ORDER = [
    ("queries_socialiqa", 1, 1),
    ("queries_mmlu_social_science", 1, 2),
    ("queries_arc_challenge", 2, 1),
    ("queries_mmlu_stem", 2, 2),
]

_PANEL_COLORAXIS_IDX = {(1, 1): 1, (1, 2): 2, (2, 1): 3, (2, 2): 4}

_PANEL_COLORBAR_POS = {
    1: {"x": 0.495, "y": 0.775, "len": 0.43, "thickness": 10, "xanchor": "left"},
    2: {"x": 1.01, "y": 0.775, "len": 0.43, "thickness": 10, "xanchor": "left"},
    3: {"x": 0.495, "y": 0.225, "len": 0.43, "thickness": 10, "xanchor": "left"},
    4: {"x": 1.01, "y": 0.225, "len": 0.43, "thickness": 10, "xanchor": "left"},
}

FIGURE_WIDTH_GRID_PX = 2000
FIGURE_HEIGHT_GRID_PX = 1600


def _zscore_matrix(mat: np.ndarray) -> np.ndarray:
    """Standardize a bin matrix across its finite cells (display units)."""
    finite = np.isfinite(mat)
    if not finite.any():
        return mat
    vals = mat[finite]
    mu = float(vals.mean())
    sigma = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
    if sigma == 0 or not np.isfinite(sigma):
        out = np.full_like(mat, np.nan)
        out[finite] = 0.0
        return out
    return (mat - mu) / sigma


def _nonsignificant_mask(
    mean_mat: np.ndarray,
    std_mat: np.ndarray,
    n_mat: np.ndarray,
    alpha: float = 0.05,
) -> np.ndarray:
    """Cells whose within-bin document mean is NOT distinguishable from zero.

    Two-sided normal-approximation test on the per-bin mean over the bin's
    documents (SE = std / sqrt(n_docs), valid by the CLT at n_docs ~ 1e4),
    with a Benjamini-Hochberg correction across the finite cells. Returns
    True where the bin is *not* significant -- the cells we flag on the map.
    """
    erfc = np.vectorize(math.erfc)
    with np.errstate(divide="ignore", invalid="ignore"):
        se = std_mat / np.sqrt(n_mat)
        tstat = np.abs(mean_mat) / se
        pvals = erfc(tstat / math.sqrt(2.0))  # two-sided normal approx
    finite = np.isfinite(pvals) & np.isfinite(mean_mat) & np.isfinite(se) & (se > 0)
    flat = np.sort(pvals[finite])
    m = int(flat.size)
    thresh = 0.0
    if m:
        passed = flat <= (np.arange(1, m + 1) / m) * alpha
        if passed.any():
            thresh = float(flat[np.nonzero(passed)[0].max()])
    return finite & (pvals > thresh)


def fig_influence_signed_2x2(
    grids: dict[str, pd.DataFrame],
    output_dir: Path,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> None:
    from dolma.distribution_report.sampling_loader import CANONICAL_TOPIC_LABELS, CANONICAL_FORMAT_LABELS
    topic_order = list(CANONICAL_TOPIC_LABELS)
    format_order = list(CANONICAL_FORMAT_LABELS)
    x_labels = [short_label(f) for f in format_order]
    y_labels = [short_label(t) for t in topic_order]

    # Display the per-bin MEAN influence (matching the "Mean influence"
    # colorbar label), standardized across bins for readable O(1) units.
    # Flag cells whose document-level mean is not significantly nonzero so
    # the visible pattern reads as estimated signal, not color intensity.
    matrices: dict[str, np.ndarray] = {}
    nonsig_masks: dict[str, np.ndarray] = {}
    for key, _, _ in _PANEL_ORDER:
        if key not in grids:
            continue
        grid = grids[key]
        raw_mean = _pivot_matrix(grid, "mean_influence", topic_order, format_order)
        std_mat = _pivot_matrix(grid, "std_score", topic_order, format_order)
        n_mat = _pivot_matrix(grid, "doc_count", topic_order, format_order)
        matrices[key] = _zscore_matrix(raw_mean) * INFLUENCE_SCALE_FACTOR
        nonsig_masks[key] = _nonsignificant_mask(raw_mean, std_mat, n_mat)

    if not matrices:
        return

    # One shared scale across all four panels keeps the cross-benchmark
    # comparison honest and prevents per-panel colorbar drift.
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
    subplot_titles = [
        subplot_title_by_cell.get((1, 1), ""),
        subplot_title_by_cell.get((1, 2), ""),
        "",
        subplot_title_by_cell.get((2, 1), ""),
        subplot_title_by_cell.get((2, 2), ""),
    ]
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[*subplot_titles, ""],
        specs=[
            [{}, {}, {"rowspan": 2}],
            [{}, {}, None],
        ],
        column_widths=[0.47, 0.47, 0.035],
        horizontal_spacing=0.055,
        vertical_spacing=0.03,
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
        mask = nonsig_masks.get(key)
        if mask is not None and bool(np.any(mask)):
            ys, xs = np.nonzero(mask)
            fig.add_trace(
                go.Scatter(
                    x=[x_labels[c] for c in xs],
                    y=[y_labels[r] for r in ys],
                    mode="markers",
                    marker={
                        "symbol": "x",
                        "size": 5,
                        "color": "rgba(25,25,25,0.9)",
                        "line": {"width": 0.6, "color": "rgba(255,255,255,0.9)"},
                    },
                    hoverinfo="skip",
                    showlegend=False,
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

    # No in-image title — the LaTeX caption carries it.
    base_layout = paper_layout(title=None, plot_bgcolor=COLOR_EMPTY_CELL)
    fig.update_layout(
        **base_layout,
        width=FIGURE_WIDTH_GRID_PX, height=FIGURE_HEIGHT_GRID_PX,
    )
    fig.update_layout(margin={"l": 200, "r": 210, "t": 30, "b": 160})
    fig.update_annotations(font=plotly_font("SUBPLOT_TITLE", scale=0.6))
    apply_heatmap_axes(fig, x_labels=x_labels, y_labels=y_labels, tick_scale=0.6)
    fig.update_xaxes(showticklabels=False, row=1)
    fig.update_yaxes(showticklabels=False, col=2)
    fig.update_xaxes(
        showticklabels=False,
        ticks="",
        showline=False,
        row=1,
        col=3,
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
        col=3,
    )
    fig.add_annotation(
        text="Format",
        x=0.5,
        y=-0.10,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=plotly_font("AXIS_TITLE", scale=0.6),
    )
    fig.add_annotation(
        text="Topic",
        x=-0.11,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        textangle=-90,
        font=plotly_font("AXIS_TITLE", scale=0.6),
    )
    fig.add_annotation(
        text="Mean influence (z-score)",
        x=1.08,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        textangle=-90,
        font=plotly_font("COLORBAR_TICK", scale=0.9),
    )
    save_figure(fig, output_dir, "fig_influence_signed_2x2", formats, FIGURE_WIDTH_GRID_PX, FIGURE_HEIGHT_GRID_PX)


__all__ = [
    "FIGURE_HEIGHT_GRID_PX",
    "FIGURE_WIDTH_GRID_PX",
    "_PANEL_ORDER",
    "fig_influence_signed_2x2",
]
