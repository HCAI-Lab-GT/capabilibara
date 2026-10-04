"""Side-by-side contrastive influence heatmaps for benchmark comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dolma.distribution_report.data_loader import display_label
from dolma.distribution_report.influence_figures import (
    _benchmark_display,
    _pivot_matrix,
)
from dolma.distribution_report.influence_loader import influence_axis_order
from dolma.distribution_report.style import (
    COLOR_EMPTY_CELL,
    DIVERGING_COLORSCALE,
    FIGURE_HEIGHT_HEATMAP_PX,
    FIGURE_HEIGHT_SIDE_BY_SIDE_PX,
    FIGURE_WIDTH_HEATMAP_PX,
    FIGURE_WIDTH_SIDE_BY_SIDE_PX,
    paper_layout,
    save_figure,
)


def fig_influence_side_by_side(
    grid_a: pd.DataFrame,
    grid_b: pd.DataFrame,
    key_a: str,
    key_b: str,
    value_col: str,
    output_dir: Path,
    formats: tuple[str, ...] = ("pdf", "png"),
    axis_order: tuple[list[str], list[str]] | None = None,
    suffix: str = "",
) -> None:
    topic_order, format_order = axis_order or influence_axis_order(grid_a)
    values_a = _pivot_matrix(grid_a, value_col, topic_order, format_order)
    values_b = _pivot_matrix(grid_b, value_col, topic_order, format_order)
    x_labels = [display_label(f) for f in format_order]
    y_labels = [display_label(t) for t in topic_order]
    name_a = _benchmark_display(key_a)
    name_b = _benchmark_display(key_b)

    is_signed = value_col == "mean_score"

    def _panel_coloraxis(values: np.ndarray, idx: int) -> dict:
        finite = values[np.isfinite(values)].ravel()
        if is_signed:
            bound = float(max(np.percentile(np.abs(finite), 95), 1e-8))
            return {
                "colorscale": DIVERGING_COLORSCALE,
                "cmid": 0,
                "cmin": -bound,
                "cmax": bound,
                "colorbar": {"title": "Mean Influence", "x": 1.0 if idx == 2 else -0.15},
            }
        return {
            "colorscale": "Viridis",
            "cmin": 0,
            "cmax": float(finite.max()),
            "colorbar": {"title": "Mean |Influence|", "x": 1.0 if idx == 2 else -0.15},
        }

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[name_a, name_b],
        horizontal_spacing=0.18,
    )
    for col_idx, (values, name) in enumerate(
        [(values_a, name_a), (values_b, name_b)], start=1
    ):
        axis_name = f"coloraxis{col_idx}"
        fig.add_trace(
            go.Heatmap(
                z=values,
                x=x_labels,
                y=y_labels,
                coloraxis=axis_name,
                xgap=1,
                ygap=1,
                name=name,
            ),
            row=1,
            col=col_idx,
        )

    kind = "signed" if is_signed else "abs"
    short_a = key_a.removeprefix("queries_")
    short_b = key_b.removeprefix("queries_")

    fig.update_layout(
        **paper_layout(
            f"{name_a} vs {name_b}: Influence by Topic and Format",
            plot_bgcolor=COLOR_EMPTY_CELL,
        ),
        coloraxis1=_panel_coloraxis(values_a, 1),
        coloraxis2=_panel_coloraxis(values_b, 2),
        width=FIGURE_WIDTH_SIDE_BY_SIDE_PX,
        height=FIGURE_HEIGHT_SIDE_BY_SIDE_PX,
    )
    fig.update_xaxes(tickangle=45)
    save_figure(
        fig,
        output_dir,
        f"fig_influence_compare_{kind}_{short_a}_vs_{short_b}{suffix}",
        formats,
        FIGURE_WIDTH_SIDE_BY_SIDE_PX,
        FIGURE_HEIGHT_SIDE_BY_SIDE_PX,
    )


def fig_influence_difference(
    grid_a: pd.DataFrame,
    grid_b: pd.DataFrame,
    key_a: str,
    key_b: str,
    output_dir: Path,
    formats: tuple[str, ...] = ("pdf", "png"),
    axis_order: tuple[list[str], list[str]] | None = None,
    suffix: str = "",
) -> None:
    topic_order, format_order = axis_order or influence_axis_order(grid_a)
    values_a = _pivot_matrix(grid_a, "mean_score", topic_order, format_order)
    values_b = _pivot_matrix(grid_b, "mean_score", topic_order, format_order)
    diff = values_a - values_b
    x_labels = [display_label(f) for f in format_order]
    y_labels = [display_label(t) for t in topic_order]
    name_a = _benchmark_display(key_a)
    name_b = _benchmark_display(key_b)

    finite = diff[np.isfinite(diff)]
    abs_max = float(max(np.percentile(np.abs(finite), 95), 1e-8))

    fig = go.Figure(
        go.Heatmap(
            z=diff,
            x=x_labels,
            y=y_labels,
            colorscale=DIVERGING_COLORSCALE,
            zmid=0,
            zmin=-abs_max,
            zmax=abs_max,
            xgap=1,
            ygap=1,
            colorbar={"title": f"{name_a} - {name_b}"},
        )
    )
    short_a = key_a.removeprefix("queries_")
    short_b = key_b.removeprefix("queries_")
    fig.update_layout(
        **paper_layout(
            f"Differential Influence: {name_a} vs {name_b}",
            plot_bgcolor=COLOR_EMPTY_CELL,
        ),
        xaxis={"tickangle": 45},
        width=FIGURE_WIDTH_HEATMAP_PX,
        height=FIGURE_HEIGHT_HEATMAP_PX,
    )
    save_figure(
        fig,
        output_dir,
        f"fig_influence_diff_{short_a}_vs_{short_b}{suffix}",
        formats,
        FIGURE_WIDTH_HEATMAP_PX,
        FIGURE_HEIGHT_HEATMAP_PX,
    )


__all__ = [
    "fig_influence_difference",
    "fig_influence_side_by_side",
]
