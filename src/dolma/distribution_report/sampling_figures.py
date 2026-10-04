"""Stratified vs representative sampling comparison figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dolma.distribution_report.data_loader import short_label
from dolma.distribution_report.sampling_data import (
    build_share_difference_table,
    comparison_matrices,
)
from dolma.distribution_report.style import (
    DIVERGING_COLORSCALE,
    FIGURE_HEIGHT_HEATMAP_PX,
    FIGURE_HEIGHT_SIDE_BY_SIDE_PX,
    FIGURE_WIDTH_HEATMAP_PX,
    FIGURE_WIDTH_SIDE_BY_SIDE_PX,
    diverging_colorbar_ticks,
    paper_layout,
    plotly_font,
    save_figure,
)


def _log_colorbar_ticks(log_min: float, log_max: float) -> tuple[list[int], list[str]]:
    tick_vals = list(range(int(np.floor(log_min)), int(np.ceil(log_max)) + 1))
    tick_text = [f"{10**v:,.0f}" for v in tick_vals]
    return tick_vals, tick_text


def fig_side_by_side(
    stratified_grid: pd.DataFrame,
    representative_grid: pd.DataFrame,
    output_dir: Path,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> None:
    strat_matrix, rep_matrix, topic_order, format_order = comparison_matrices(
        stratified_grid,
        representative_grid,
    )

    all_values = np.concatenate(
        [strat_matrix.values.ravel(), rep_matrix.values.ravel()]
    )
    if not (all_values > 0).any():
        return

    # Color range bounded by the actual non-zero data; zero-token cells get
    # NaN'd in the per-panel loop so they render as plot background (white)
    # rather than competing with real low-token cells for the dark end of
    # the Viridis scale.
    nonzero = all_values[all_values > 0]
    log_min = float(np.log10(nonzero.min())) if len(nonzero) else 0.0
    log_max = float(np.log10(nonzero.max())) if len(nonzero) else 1.0
    tick_vals, tick_text = _log_colorbar_ticks(log_min, log_max)

    topic_labels = [short_label(t) for t in topic_order]
    format_labels = [short_label(f) for f in format_order]

    # Round 8: vertical 2x1 stack with (a)/(b) sub-captions (user feedback:
    # "the heat maps are not square... put them vertically on top not
    # horizontally" AND "we need to say figure 9a and 9b which we don't
    # currently do").
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=["<b>(a) Stratified sample</b>",
                        "<b>(b) Representative sample</b>"],
        vertical_spacing=0.12,
    )

    for row_idx, (matrix, name) in enumerate(
        [(strat_matrix, "Stratified"), (rep_matrix, "Representative")], start=1
    ):
        vals = matrix.values.astype(float)
        # True zero-token cells → NaN so Plotly renders them as the white
        # plot background (visually empty), rather than at the Viridis min
        # color which would otherwise paint a wall of dark purple across
        # the sparse representative-sample panel.
        log_z = np.where(vals > 0, np.log10(np.maximum(vals, 1e-9)), np.nan)
        fig.add_trace(
            go.Heatmap(
                z=log_z,
                x=format_labels,
                y=topic_labels,
                coloraxis="coloraxis",
                # Round 8: gray xgap/ygap so empty cells inside the grid
                # show as light gray boxes rather than pure white merging
                # into the background (user feedback: "we still wanna
                # have some way to visualize the grid and the emptiness
                # of it").
                xgap=1,
                ygap=1,
                name=name,
                hoverongaps=False,
            ),
            row=row_idx,
            col=1,
        )

    # Round 8: faint gray plot bg so empty cells (NaN) show as
    # distinguishable boxes inside the grid; white was too washy.
    fig.update_layout(
        **paper_layout(title=None, plot_bgcolor="#eaeaea"),
        coloraxis={
            "colorscale": "Viridis",
            "colorbar": {
                "title": "Tokens (log scale)",
                "tickfont": plotly_font("COLORBAR_TICK"),
                "tickmode": "array",
                "tickvals": tick_vals,
                "ticktext": tick_text,
                "len": 1.0, "y": 0.5, "yanchor": "middle", "thickness": 18,
            },
            "cmin": log_min,
            "cmax": log_max,
        },
        # Round 8: tall 2x1 stack at heatmap width.
        width=FIGURE_WIDTH_HEATMAP_PX,
        height=int(FIGURE_HEIGHT_HEATMAP_PX * 1.9),
    )
    fig.update_xaxes(tickangle=45, tickfont=plotly_font("TICK"))
    fig.update_yaxes(tickfont=plotly_font("TICK"))
    fig.update_annotations(font=plotly_font("SUBPLOT_TITLE"))
    fig.layout.margin.b = 130
    fig.add_annotation(
        text="Format",
        x=0.5, y=-0.09, xref="paper", yref="paper", showarrow=False,
        font=plotly_font("AXIS_TITLE"),
    )
    fig.add_annotation(
        text="Topic",
        x=-0.07, y=0.5, xref="paper", yref="paper", showarrow=False,
        textangle=-90, font=plotly_font("AXIS_TITLE"),
    )
    save_figure(
        fig,
        output_dir,
        "fig_stratified_vs_representative",
        formats,
        FIGURE_WIDTH_SIDE_BY_SIDE_PX,
        FIGURE_HEIGHT_SIDE_BY_SIDE_PX,
    )


def fig_difference(
    stratified_grid: pd.DataFrame,
    representative_grid: pd.DataFrame,
    output_dir: Path,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> None:
    diff_df = build_share_difference_table(stratified_grid, representative_grid)
    _, _, topic_order, format_order = comparison_matrices(
        stratified_grid,
        representative_grid,
    )
    log_ratio = (
        diff_df["log2_share_ratio"]
        .to_numpy()
        .reshape(len(topic_order), len(format_order))
    )
    finite = log_ratio[np.isfinite(log_ratio)]
    if len(finite) == 0:
        return
    topic_labels = [short_label(t) for t in topic_order]
    format_labels = [short_label(f) for f in format_order]
    abs_max = max(abs(finite.min()), abs(finite.max()), 0.1)
    tick_vals, tick_text = diverging_colorbar_ticks(abs_max, n_ticks=5, decimals=2)

    fig = go.Figure(
        go.Heatmap(
            z=log_ratio,
            x=format_labels,
            y=topic_labels,
            colorscale=DIVERGING_COLORSCALE,
            zmid=0,
            zmin=-abs_max,
            zmax=abs_max,
            xgap=1,
            ygap=1,
            colorbar={
                "title": "log₂(stratified / representative)",
                "tickfont": plotly_font("COLORBAR_TICK"),
            },
        )
    )
    fig.update_layout(
        **paper_layout(title=None, plot_bgcolor="white"),
        xaxis={"tickangle": 45, "tickfont": plotly_font("TICK")},
        yaxis={"tickfont": plotly_font("TICK")},
        width=FIGURE_WIDTH_HEATMAP_PX,
        height=FIGURE_HEIGHT_HEATMAP_PX,
    )
    fig.update_xaxes(title_text="Format", title_font=plotly_font("AXIS_TITLE"))
    fig.update_yaxes(title_text="Topic", title_font=plotly_font("AXIS_TITLE"))
    save_figure(
        fig,
        output_dir,
        "fig_sampling_difference",
        formats,
        FIGURE_WIDTH_HEATMAP_PX,
        FIGURE_HEIGHT_HEATMAP_PX,
    )


__all__ = [
    "fig_difference",
    "fig_side_by_side",
]
