"""Marginal influence bar charts collapsed from the 24x24 bin grid."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd
import plotly.graph_objects as go

from dolma.distribution_report.data_loader import short_label
from dolma.distribution_report.influence_figures import _benchmark_display
from dolma.distribution_report.style import (
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    FIGURE_HEIGHT_BAR_PX,
    FIGURE_WIDTH_PX,
    paper_layout,
    plotly_font,
    save_figure,
)

_PAIRED_BAR_WIDTH_PX = 1200
_PAIRED_BAR_HEIGHT_PX = 820


def _marginal_df(
    grid_df: pd.DataFrame,
    group_col: str,
    value_col: str,
) -> pd.DataFrame:
    agg = grid_df.groupby(group_col, as_index=False).agg(
        value=(value_col, "mean"),
        doc_count=("doc_count", "sum"),
    )
    return cast(Any, agg).sort_values("value")


def _influence_bar(
    grid_df: pd.DataFrame,
    group_col: str,
    value_col: str,
    title: str,
    xlabel: str,
    output_dir: Path,
    filename: str,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> None:
    mdf = _marginal_df(grid_df, group_col, value_col)
    labels = [short_label(lbl) for lbl in mdf[group_col]]
    values = mdf["value"].tolist()

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=COLOR_PRIMARY,
        )
    )
    layout = paper_layout(title=None)
    layout["margin"] = {"l": 250, "r": 280, "t": 35, "b": 105}
    fig.update_layout(
        **layout,
        xaxis={"title": xlabel,
                   "title_font": plotly_font("AXIS_TITLE"),
                   "tickfont": plotly_font("TICK")},
        yaxis={"title": "", "tickfont": plotly_font("TICK")},
        width=FIGURE_WIDTH_PX,
        height=FIGURE_HEIGHT_BAR_PX,
    )
    save_figure(
        fig, output_dir, filename, formats, FIGURE_WIDTH_PX, FIGURE_HEIGHT_BAR_PX
    )


def fig_influence_topic_bars(
    grid_df: pd.DataFrame,
    benchmark_key: str,
    output_dir: Path,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> None:
    name = _benchmark_display(benchmark_key)
    short = benchmark_key.removeprefix("queries_")
    _influence_bar(
        grid_df,
        group_col="topic_label",
        value_col="mean_score",
        title=f"{name}: Mean Influence by Topic",
        xlabel="Mean Influence Score",
        output_dir=output_dir,
        filename=f"fig_influence_topic_{short}",
        formats=formats,
    )


def fig_influence_format_bars(
    grid_df: pd.DataFrame,
    benchmark_key: str,
    output_dir: Path,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> None:
    name = _benchmark_display(benchmark_key)
    short = benchmark_key.removeprefix("queries_")
    _influence_bar(
        grid_df,
        group_col="format_label",
        value_col="mean_score",
        title=f"{name}: Mean Influence by Format",
        xlabel="Mean Influence Score",
        output_dir=output_dir,
        filename=f"fig_influence_format_{short}",
        formats=formats,
    )


# Round 8: semantic color palette matching influence_composite_signed_topic.
# Purple = social benchmarks, green = reasoning/STEM. Same hex codes as
# _PANEL_STYLE in that module — keeps the cross-figure visual story
# (user feedback Round 8: "we want to unify them with examples like
# figure 14 ... it would be weird to me to have Purple and green
# being like social and science and then here we're not using purple
# and Green again").
_BENCH_COLORS = {
    "queries_socialiqa":            "#984EA3",  # purple (social reasoning)
    "queries_mmlu_social_science":  "#984EA3",  # purple (social knowledge)
    "queries_arc_challenge":        "#1b7837",  # green (reasoning)
    "queries_mmlu_stem":            "#1b7837",  # green (STEM knowledge)
    "queries_arc_easy":             "#66c2a4",  # lighter green
    "queries_gsm8k":                "#1b7837",  # green
    "queries_bbh_snarks":           "#e6550d",  # orange (held-out)
}


def fig_influence_topic_paired(
    grid_a: pd.DataFrame,
    grid_b: pd.DataFrame,
    key_a: str,
    key_b: str,
    output_dir: Path,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> None:
    mdf_a = _marginal_df(grid_a, "topic_label", "mean_score")
    mdf_b = _marginal_df(grid_b, "topic_label", "mean_score")
    merged = mdf_a.merge(
        mdf_b, on="topic_label", suffixes=("_a", "_b"), how="outer"
    ).fillna(0)
    merged = merged.sort_values("value_a", ascending=True)
    labels = [short_label(lbl) for lbl in merged["topic_label"]]
    name_a = _benchmark_display(key_a)
    name_b = _benchmark_display(key_b)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=merged["value_a"].tolist(),
            y=labels,
            orientation="h",
            name=name_a,
            marker_color=_BENCH_COLORS.get(key_a, COLOR_PRIMARY),
        )
    )
    fig.add_trace(
        go.Bar(
            x=merged["value_b"].tolist(),
            y=labels,
            orientation="h",
            name=name_b,
            marker_color=_BENCH_COLORS.get(key_b, COLOR_SECONDARY),
        )
    )
    short_a = key_a.removeprefix("queries_")
    short_b = key_b.removeprefix("queries_")
    fig.update_layout(
        **paper_layout(title=None),
        barmode="group",
        bargap=0.22,
        xaxis={"title": "Mean influence score (z-score)",
                   "title_font": plotly_font("AXIS_TITLE", scale=0.85),
                   "tickfont": plotly_font("TICK", scale=0.85),
                   "zeroline": True,
                   "zerolinecolor": "#222222",
                   "zerolinewidth": 1},
        yaxis={"title": "",
                   "tickmode": "array",
                   "tickvals": labels,
                   "ticktext": labels,
                   "tickfont": plotly_font("TICK", scale=0.78),
                   "automargin": True},
        legend={
            "font": plotly_font("LEGEND", scale=0.75),
            "x": 1.02,
            "xanchor": "left",
            "y": 0.98,
            "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.85)",
            "borderwidth": 0,
        },
        width=_PAIRED_BAR_WIDTH_PX,
        height=_PAIRED_BAR_HEIGHT_PX,
    )
    save_figure(
        fig,
        output_dir,
        f"fig_influence_topic_paired_{short_a}_vs_{short_b}",
        formats,
        _PAIRED_BAR_WIDTH_PX,
        _PAIRED_BAR_HEIGHT_PX,
    )


__all__ = [
    "fig_influence_format_bars",
    "fig_influence_topic_bars",
    "fig_influence_topic_paired",
]
