"""24x24 heatmaps for joint topic-format distributions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dolma.distribution_report.data_loader import short_label
from dolma.distribution_report.style import (
    COLOR_EMPTY_CELL,
    FIGURE_HEIGHT_HEATMAP_PX,
    FIGURE_WIDTH_HEATMAP_PX,
    apply_colorbar_strip_axes,
    apply_heatmap_axes,
    paper_layout,
    plotly_font,
    save_figure,
    sequential_colorbar_strip,
)


def _build_matrix(
    grid_df: pd.DataFrame,
    value_col: str,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    from dolma.distribution_report.sampling_loader import CANONICAL_TOPIC_LABELS, CANONICAL_FORMAT_LABELS
    topic_order = list(CANONICAL_TOPIC_LABELS)
    format_order = list(CANONICAL_FORMAT_LABELS)

    matrix = grid_df.pivot(
        index="topic_label", columns="format_label", values=value_col
    ).fillna(0)
    matrix = matrix.reindex(index=topic_order, columns=format_order, fill_value=0)
    return matrix, topic_order, format_order


def _format_count_tick(value: float) -> str:
    return f"{int(round(value)):,.0f}"


def _log_colorbar_ticks(
    log_min: float,
    log_max: float,
) -> tuple[list[float], list[str]]:
    """Return readable log-count ticks using 1/2/5 steps per decade."""
    if not np.isfinite(log_min) or not np.isfinite(log_max):
        return [0.0], ["1"]
    if log_max < log_min:
        log_min, log_max = log_max, log_min

    candidates: list[float] = []
    min_exp = int(np.floor(log_min))
    max_exp = int(np.ceil(log_max))
    for exponent in range(min_exp, max_exp + 1):
        for mantissa in (1, 2, 5):
            value = mantissa * (10 ** exponent)
            log_value = float(np.log10(value))
            if log_min <= log_value <= log_max:
                candidates.append(log_value)

    if not candidates:
        value = 10 ** log_min
        return [float(log_min)], [_format_count_tick(value)]

    # Keep labels dense enough to show within-decade structure, but not so
    # dense that long comma-formatted counts collide on short colorbars.
    if len(candidates) > 8:
        candidates = [
            log_value
            for log_value in candidates
            if int(round(10 ** (log_value - np.floor(log_value)))) in (1, 5)
        ]
    tick_text = [_format_count_tick(10 ** v) for v in candidates]
    tick_vals = [float(v) for v in candidates]
    return tick_vals, tick_text


def _log_coloraxis(
    log_z: np.ndarray,
    title: str,
    *,
    length: float,
    y: float,
    x: float | None = None,
    xanchor: str | None = None,
) -> dict:
    log_min = float(log_z.min())
    log_max = float(log_z.max())
    tick_vals, tick_text = _log_colorbar_ticks(log_min, log_max)
    return {
        "colorscale": "Viridis",
        "cmin": log_min,
        "cmax": log_max,
        "colorbar": dict(
            title={"text": title, "side": "right"},
            tickfont=plotly_font("COLORBAR_TICK"),
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_text,
            ticks="outside",
            ticklen=4,
            tickwidth=1,
            len=length,
            y=y,
            yanchor="middle",
            thickness=22,
            outlinewidth=0,
            **({} if x is None else {"x": x}),
            **({} if xanchor is None else {"xanchor": xanchor}),
        ),
    }


def _heatmap(
    grid_df: pd.DataFrame,
    value_col: str,
    title: str,
    cbar_label: str,
    output_dir: Path,
    filename: str,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> None:
    matrix, topic_order, format_order = _build_matrix(grid_df, value_col)
    values = matrix.values.astype(float)

    # Floor zero-doc cells to 1 (log10=0) so they render at the minimum color
    # of the scale instead of NaN/grey. Every bin has at least 1 doc in the
    # final corpus; the floor is purely for display contiguity.
    log_z = np.log10(np.maximum(values, 1))
    x_labels = [short_label(f) for f in format_order]
    y_labels = [short_label(t) for t in topic_order]

    z_min = float(log_z.min())
    z_max = float(log_z.max())
    tick_vals, tick_text = _log_colorbar_ticks(z_min, z_max)
    strip, tick_vals, tick_text = sequential_colorbar_strip(
        z_min,
        z_max,
        tick_vals=tick_vals,
        tick_text=tick_text,
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[f"<b>{title}</b>", ""],
        column_widths=[0.92, 0.04],
        horizontal_spacing=0.045,
    )
    fig.add_trace(
        go.Heatmap(
            z=log_z,
            x=x_labels,
            y=y_labels,
            colorscale="Viridis",
            zmin=z_min,
            zmax=z_max,
            showscale=False,
            xgap=1,
            ygap=1,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=strip,
            x=[""],
            y=np.linspace(z_min, z_max, strip.shape[0]),
            colorscale="Viridis",
            zmin=z_min,
            zmax=z_max,
            showscale=False,
            xgap=0,
            ygap=0,
        ),
        row=1,
        col=2,
    )

    layout = paper_layout(title=None, plot_bgcolor=COLOR_EMPTY_CELL)
    layout["margin"] = {"l": 170, "r": 180, "t": 70, "b": 145}
    fig.update_layout(
        **layout,
        width=FIGURE_WIDTH_HEATMAP_PX,
        height=FIGURE_HEIGHT_HEATMAP_PX,
    )
    fig.update_annotations(font=plotly_font("SUBPLOT_TITLE", scale=0.9))
    apply_heatmap_axes(fig, x_labels=x_labels, y_labels=y_labels, tick_scale=0.75)
    fig.update_xaxes(
        title_text="Format",
        title_font=plotly_font("AXIS_TITLE", scale=0.9),
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="Topic",
        title_font=plotly_font("AXIS_TITLE", scale=0.9),
        row=1,
        col=1,
    )
    apply_colorbar_strip_axes(
        fig,
        row=1,
        col=2,
        tick_vals=tick_vals,
        tick_text=tick_text,
        title=cbar_label,
        title_x=1.08,
        title_y=0.5,
    )
    save_figure(
        fig,
        output_dir,
        filename,
        formats,
        FIGURE_WIDTH_HEATMAP_PX,
        FIGURE_HEIGHT_HEATMAP_PX,
    )


def fig_heatmap_doc_count(
    grid_df: pd.DataFrame,
    output_dir: Path,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> None:
    _heatmap(
        grid_df,
        value_col="doc_count",
        title="(a) Document counts",
        cbar_label="Documents (log scale)",
        output_dir=output_dir,
        filename="fig_heatmap_doc_count",
        formats=formats,
    )


def fig_heatmap_token_count(
    grid_df: pd.DataFrame,
    output_dir: Path,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> None:
    _heatmap(
        grid_df,
        value_col="token_count_est",
        title="(b) Token counts",
        cbar_label="Tokens (log scale)",
        output_dir=output_dir,
        filename="fig_heatmap_token_count",
        formats=formats,
    )


def fig_heatmap_joint_2panel(
    grid_df: pd.DataFrame,
    output_dir: Path,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> None:
    """1x2 joint topic-format heatmap: (a) doc counts | (b) token counts.

    Round 7 replacement for the previous fig_heatmap_doc_count +
    fig_heatmap_token_count pair (user flagged them as visually
    redundant: same topology, similar captions). One figure, two
    panels, two log-scale colorbars (the metrics differ by ~3 orders
    of magnitude so per-panel colorbars are honest).
    """
    matrix_d, topic_order, format_order = _build_matrix(grid_df, "doc_count")
    matrix_t, _, _ = _build_matrix(grid_df, "token_count_est")

    log_d = np.log10(np.maximum(matrix_d.values.astype(float), 1))
    log_t = np.log10(np.maximum(matrix_t.values.astype(float), 1))
    x_labels = [short_label(f) for f in format_order]
    y_labels = [short_label(t) for t in topic_order]
    d_min, d_max = float(log_d.min()), float(log_d.max())
    t_min, t_max = float(log_t.min()), float(log_t.max())
    d_ticks, d_text = _log_colorbar_ticks(d_min, d_max)
    t_ticks, t_text = _log_colorbar_ticks(t_min, t_max)
    d_strip, d_ticks, d_text = sequential_colorbar_strip(
        d_min,
        d_max,
        tick_vals=d_ticks,
        tick_text=d_text,
    )
    t_strip, t_ticks, t_text = sequential_colorbar_strip(
        t_min,
        t_max,
        tick_vals=t_ticks,
        tick_text=t_text,
    )

    # Round 8: vertical 2x1 stack instead of 1x2 horizontal (user feedback:
    # "the best way to represent these is not having this plot horizontally
    # next to each other but vertically ... they're currently scrunched up").
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["<b>(a) Document counts</b>",
                        "",
                        "<b>(b) Token counts</b>",
                        ""],
        column_widths=[0.94, 0.035],
        horizontal_spacing=0.04,
        vertical_spacing=0.12,
    )
    fig.add_trace(
        go.Heatmap(z=log_d, x=x_labels, y=y_labels,
                   colorscale="Viridis", zmin=d_min, zmax=d_max,
                   showscale=False, xgap=1, ygap=1),
        row=1, col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=d_strip,
            x=[""],
            y=np.linspace(d_min, d_max, d_strip.shape[0]),
            colorscale="Viridis",
            zmin=d_min,
            zmax=d_max,
            showscale=False,
            xgap=0,
            ygap=0,
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Heatmap(z=log_t, x=x_labels, y=y_labels,
                   colorscale="Viridis", zmin=t_min, zmax=t_max,
                   showscale=False, xgap=1, ygap=1),
        row=2, col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=t_strip,
            x=[""],
            y=np.linspace(t_min, t_max, t_strip.shape[0]),
            colorscale="Viridis",
            zmin=t_min,
            zmax=t_max,
            showscale=False,
            xgap=0,
            ygap=0,
        ),
        row=2, col=2,
    )

    base_layout = paper_layout(title=None, plot_bgcolor=COLOR_EMPTY_CELL)
    base_layout["margin"]["r"] = 210
    fig.update_layout(
        **base_layout,
        width=FIGURE_WIDTH_HEATMAP_PX,
        # Tall: each panel ~1000px high so 24x24 cells render close to
        # square on a single-column COLM page.
        height=int(FIGURE_HEIGHT_HEATMAP_PX * 1.9),
    )
    fig.update_annotations(font=plotly_font("SUBPLOT_TITLE", scale=0.8))
    apply_heatmap_axes(fig, x_labels=x_labels, y_labels=y_labels, tick_scale=0.6)
    fig.update_xaxes(title_text="Format", title_font=plotly_font("AXIS_TITLE", scale=0.8), row=2, col=1)
    fig.update_yaxes(title_text="Topic", title_font=plotly_font("AXIS_TITLE", scale=0.8), col=1)
    apply_colorbar_strip_axes(
        fig,
        row=1,
        col=2,
        tick_vals=d_ticks,
        tick_text=d_text,
        title="Documents (log scale)",
        title_x=1.08,
        title_y=0.78,
    )
    apply_colorbar_strip_axes(
        fig,
        row=2,
        col=2,
        tick_vals=t_ticks,
        tick_text=t_text,
        title="Tokens (log scale)",
        title_x=1.08,
        title_y=0.22,
    )
    save_figure(
        fig, output_dir, "fig_heatmap_joint_2panel",
        formats, FIGURE_WIDTH_HEATMAP_PX, int(FIGURE_HEIGHT_HEATMAP_PX * 1.9),
    )


__all__ = [
    "fig_heatmap_doc_count",
    "fig_heatmap_joint_2panel",
    "fig_heatmap_token_count",
]
