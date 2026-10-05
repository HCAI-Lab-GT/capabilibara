"""24x24 influence heatmaps for bin-level attribution scores."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dolma.distribution_report.data_loader import short_label
from dolma.distribution_report.influence_loader import (
    BENCHMARK_DISPLAY_NAMES,
    influence_axis_order,
)
from dolma.distribution_report.style import (
    COLOR_EMPTY_CELL,
    DIVERGING_COLORSCALE,
    FIGURE_HEIGHT_HEATMAP_PX,
    FIGURE_WIDTH_HEATMAP_PX,
    apply_colorbar_strip_axes,
    apply_heatmap_axes,
    diverging_colorbar_strip,
    paper_layout,
    plotly_font,
    save_figure,
    symmetric_color_limit,
)


def _pivot_matrix(
    grid_df: pd.DataFrame,
    value_col: str,
    topic_order: list[str],
    format_order: list[str],
) -> np.ndarray:
    matrix = grid_df.pivot(
        index="topic_label", columns="format_label", values=value_col
    ).fillna(0)
    matrix = matrix.reindex(index=topic_order, columns=format_order, fill_value=0)
    return matrix.values.astype(float)


def _benchmark_display(key: str) -> str:
    return BENCHMARK_DISPLAY_NAMES.get(
        key, key.replace("queries_", "").replace("_", " ").title()
    )


def fig_influence_absolute(
    grid_df: pd.DataFrame,
    benchmark_key: str,
    output_dir: Path,
    formats: tuple[str, ...] = ("pdf", "png"),
    axis_order: tuple[list[str], list[str]] | None = None,
    suffix: str = "",
) -> None:
    topic_order, format_order = axis_order or influence_axis_order(grid_df)
    values = _pivot_matrix(grid_df, "abs_mean_score", topic_order, format_order)
    x_labels = [short_label(f) for f in format_order]
    y_labels = [short_label(t) for t in topic_order]
    short = benchmark_key.removeprefix("queries_")

    fig = go.Figure(
        go.Heatmap(
            z=values,
            x=x_labels,
            y=y_labels,
            colorscale="Viridis",
            zmin=0,
            xgap=1,
            ygap=1,
            colorbar={"title": "Mean |Influence|",
                           "tickfont": plotly_font("COLORBAR_TICK")},
        )
    )
    fig.update_layout(
        **paper_layout(title=None, plot_bgcolor=COLOR_EMPTY_CELL),
        xaxis={"tickangle": 45, "tickfont": plotly_font("TICK")},
        yaxis={"tickfont": plotly_font("TICK")},
        width=FIGURE_WIDTH_HEATMAP_PX,
        height=FIGURE_HEIGHT_HEATMAP_PX,
    )
    save_figure(
        fig,
        output_dir,
        f"fig_influence_abs_{short}{suffix}",
        formats,
        FIGURE_WIDTH_HEATMAP_PX,
        FIGURE_HEIGHT_HEATMAP_PX,
    )


def fig_influence_signed(
    grid_df: pd.DataFrame,
    benchmark_key: str,
    output_dir: Path,
    formats: tuple[str, ...] = ("pdf", "png"),
    axis_order: tuple[list[str], list[str]] | None = None,
    suffix: str = "",
) -> None:
    topic_order, format_order = axis_order or influence_axis_order(grid_df)
    values = _pivot_matrix(grid_df, "mean_score", topic_order, format_order)
    x_labels = [short_label(f) for f in format_order]
    y_labels = [short_label(t) for t in topic_order]
    short = benchmark_key.removeprefix("queries_")

    finite = values[np.isfinite(values)]
    abs_max = symmetric_color_limit(finite)
    colorbar_values, colorbar_ticks, colorbar_text = diverging_colorbar_strip(
        abs_max,
        n_ticks=7,
        decimals=1,
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.92, 0.04],
        horizontal_spacing=0.04,
    )
    fig.add_trace(
        go.Heatmap(
            z=values,
            x=x_labels,
            y=y_labels,
            colorscale=DIVERGING_COLORSCALE,
            zmid=0,
            zmin=-abs_max,
            zmax=abs_max,
            xgap=1,
            ygap=1,
            showscale=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=colorbar_values,
            x=[""],
            y=np.linspace(-abs_max, abs_max, colorbar_values.shape[0]),
            colorscale=DIVERGING_COLORSCALE,
            zmid=0,
            zmin=-abs_max,
            zmax=abs_max,
            showscale=False,
            xgap=0,
            ygap=0,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        **paper_layout(title=None, plot_bgcolor=COLOR_EMPTY_CELL),
        width=FIGURE_WIDTH_HEATMAP_PX,
        height=FIGURE_HEIGHT_HEATMAP_PX,
    )
    fig.update_layout(margin={"l": 170, "r": 180, "t": 30, "b": 140})
    apply_heatmap_axes(fig, x_labels=x_labels, y_labels=y_labels, tick_scale=0.75)
    fig.update_xaxes(title_text="Format", title_font=plotly_font("AXIS_TITLE", scale=0.9), row=1, col=1)
    fig.update_yaxes(title_text="Topic", title_font=plotly_font("AXIS_TITLE", scale=0.9), row=1, col=1)
    apply_colorbar_strip_axes(
        fig,
        row=1,
        col=2,
        tick_vals=colorbar_ticks,
        tick_text=colorbar_text,
        title="Mean Influence (z-score)",
        title_x=1.08,
        title_y=0.5,
    )
    save_figure(
        fig,
        output_dir,
        f"fig_influence_signed_{short}{suffix}",
        formats,
        FIGURE_WIDTH_HEATMAP_PX,
        FIGURE_HEIGHT_HEATMAP_PX,
    )


__all__ = [
    "_benchmark_display",
    "_pivot_matrix",
    "fig_influence_absolute",
    "fig_influence_signed",
]
