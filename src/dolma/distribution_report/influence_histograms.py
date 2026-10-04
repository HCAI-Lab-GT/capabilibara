"""Score distribution histograms from bin-level statistics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dolma.distribution_report.influence_figures import _benchmark_display
from plotly.subplots import make_subplots

from dolma.distribution_report.style import (
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    FIGURE_HEIGHT_BAR_PX,
    FIGURE_WIDTH_PX,
    paper_layout,
    plotly_font,
    save_figure,
)

_HIST_COLORS = [
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    "#009E73",
    "#E69F00",
]

N_SAMPLES_PER_BIN = 200


def _sample_from_bins(grid_df: pd.DataFrame, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = []
    for _, row in grid_df.iterrows():
        n = min(int(row["doc_count"]), N_SAMPLES_PER_BIN)
        if n == 0 or row["std_score"] <= 0:
            samples.extend([row["mean_score"]] * n)
        else:
            drawn = rng.normal(row["mean_score"], row["std_score"], size=n)
            samples.extend(drawn.tolist())
    return np.array(samples)


def fig_influence_histogram(
    grid_df: pd.DataFrame,
    benchmark_key: str,
    output_dir: Path,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> None:
    samples = _sample_from_bins(grid_df)
    short = benchmark_key.removeprefix("queries_")

    fig = go.Figure(
        go.Histogram(
            x=samples,
            nbinsx=100,
            marker_color=COLOR_PRIMARY,
            opacity=0.8,
        )
    )
    fig.update_layout(
        **paper_layout(title=None),
        xaxis={"title": "Mean Influence Score (z-score)",
                   "title_font": plotly_font("AXIS_TITLE"),
                   "tickfont": plotly_font("TICK")},
        yaxis={"title": "Count (approx.)",
                   "title_font": plotly_font("AXIS_TITLE"),
                   "tickfont": plotly_font("TICK")},
        width=FIGURE_WIDTH_PX,
        height=FIGURE_HEIGHT_BAR_PX,
    )
    save_figure(
        fig,
        output_dir,
        f"fig_influence_hist_{short}",
        formats,
        FIGURE_WIDTH_PX,
        FIGURE_HEIGHT_BAR_PX,
    )


def fig_influence_histogram_overlay(
    grids: dict[str, pd.DataFrame],
    output_dir: Path,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> None:
    fig = go.Figure()
    for i, (key, grid) in enumerate(grids.items()):
        samples = _sample_from_bins(grid, seed=42 + i)
        fig.add_trace(
            go.Histogram(
                x=samples,
                nbinsx=100,
                name=_benchmark_display(key),
                marker_color=_HIST_COLORS[i % len(_HIST_COLORS)],
                opacity=0.5,
            )
        )
    fig.update_layout(
        **paper_layout(title=None),
        barmode="overlay",
        xaxis={"title": "Mean Influence Score (z-score)",
                   "title_font": plotly_font("AXIS_TITLE"),
                   "tickfont": plotly_font("TICK")},
        yaxis={"title": "Count (approx.)",
                   "title_font": plotly_font("AXIS_TITLE"),
                   "tickfont": plotly_font("TICK")},
        legend={"font": plotly_font("LEGEND")},
        width=FIGURE_WIDTH_PX,
        height=FIGURE_HEIGHT_BAR_PX,
    )
    save_figure(
        fig,
        output_dir,
        "fig_influence_hist_overlay",
        formats,
        FIGURE_WIDTH_PX,
        FIGURE_HEIGHT_BAR_PX,
    )


# Per-benchmark color convention (matches influence_composite_signed_topic).
# Reasoning benchmarks use green, knowledge benchmarks use purple-ish.
_PANEL_COLORS = {
    "queries_socialiqa":            "#984EA3",  # purple (social reasoning)
    "queries_mmlu_social_science":  "#984EA3",  # purple (social knowledge)
    "queries_arc_challenge":        "#1b7837",  # green (reasoning)
    "queries_mmlu_stem":            "#1b7837",  # green (STEM knowledge)
    "queries_bbh_snarks":           "#e6550d",  # orange (held-out)
}

# Default 2x2 panel order for fig_influence_histogram_grid; BBH Snarks is
# drawn as a thin overlay on every panel so the held-out comparison stays
# visible without consuming a separate axes.
_PRIMARY_BENCH_ORDER = [
    "queries_socialiqa",
    "queries_mmlu_social_science",
    "queries_arc_challenge",
    "queries_mmlu_stem",
]


def fig_influence_histogram_grid(
    grids: dict[str, pd.DataFrame],
    output_dir: Path,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> None:
    """4x1 per-benchmark histogram stack (Round 8 layout).

    User explicit feedback Round 8: BBH Snarks should NOT overlay these
    panels ("we're just trying to show the distributions of the scores
    for the four benchmarks"); panels should be stacked vertically with
    a whole page of room so each distribution reads clearly. So 2x2
    grid -> 4x1 vertical stack at near-full page height; no overlay.
    """
    primary = [k for k in _PRIMARY_BENCH_ORDER if k in grids]
    if len(primary) < 4:
        # Fall back to the legacy overlay if the canonical 4 aren't present.
        fig_influence_histogram_overlay(grids, output_dir, formats)
        return

    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=[f"<b>{_benchmark_display(k)}</b>" for k in primary],
        vertical_spacing=0.06,
        shared_xaxes=True,
    )

    for i, key in enumerate(primary):
        samples = _sample_from_bins(grids[key], seed=42 + i)
        fig.add_trace(
            go.Histogram(
                x=samples, nbinsx=80,
                marker_color=_PANEL_COLORS.get(key, COLOR_PRIMARY),
                opacity=0.9, showlegend=False,
            ),
            row=i + 1, col=1,
        )

    fig.update_layout(
        **paper_layout(title=None),
        width=FIGURE_WIDTH_PX,
        # Tall: ~1600px so each of the 4 stacked panels gets enough
        # vertical room to read its distribution shape clearly.
        height=int(FIGURE_HEIGHT_BAR_PX * 2.5),
    )
    fig.update_annotations(font=plotly_font("SUBPLOT_TITLE"))
    # Only the bottom-most panel gets the x-axis title; all panels get
    # tick labels via shared_xaxes (Plotly default).
    fig.update_xaxes(tickfont=plotly_font("TICK"))
    fig.update_xaxes(
        title_text="Mean Influence Score (z-score)",
        title_font=plotly_font("AXIS_TITLE"),
        row=4,
    )
    fig.update_yaxes(
        title_text="Count (approx.)",
        title_font=plotly_font("AXIS_TITLE"),
        tickfont=plotly_font("TICK"),
    )
    save_figure(
        fig,
        output_dir,
        "fig_influence_hist_overlay",  # same filename → no manuscript-tex churn
        formats,
        FIGURE_WIDTH_PX,
        int(FIGURE_HEIGHT_BAR_PX * 2.5),
    )


__all__ = [
    "fig_influence_histogram",
    "fig_influence_histogram_grid",
    "fig_influence_histogram_overlay",
]
