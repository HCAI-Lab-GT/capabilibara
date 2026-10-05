"""Horizontal bar charts for marginal topic/format distributions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from dolma.distribution_report.data_loader import short_label
from dolma.distribution_report.style import (
    COLOR_PRIMARY,
    FIGURE_HEIGHT_BAR_PX,
    paper_layout,
    plotly_font,
    save_figure,
)

_MARGINAL_BAR_WIDTH_PX = 1200
_MARGINAL_ROW_PX = 34
_MARGINAL_EXTRA_HEIGHT_PX = 220


def _bar_chart(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    title: str,
    xlabel: str,
    output_dir: Path,
    filename: str,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> None:
    sorted_df = df.sort_values(value_col, ascending=True)
    labels = [short_label(label) for label in sorted_df[label_col]]
    values = sorted_df[value_col].tolist()
    height = max(FIGURE_HEIGHT_BAR_PX, _MARGINAL_ROW_PX * len(labels) + _MARGINAL_EXTRA_HEIGHT_PX)

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=COLOR_PRIMARY,
        )
    )
    layout = paper_layout(title)
    layout["margin"] = {"l": 260, "r": 50, "t": 90, "b": 100}
    fig.update_layout(
        **layout,
        xaxis={
            "title": xlabel,
            "tickformat": ",",
            "title_font": plotly_font("AXIS_TITLE", scale=0.85),
            "tickfont": plotly_font("TICK", scale=0.85),
        },
        yaxis={
            "title": "",
            "tickmode": "array",
            "tickvals": labels,
            "ticktext": labels,
            "tickfont": plotly_font("TICK", scale=0.85),
            "automargin": True,
        },
        width=_MARGINAL_BAR_WIDTH_PX,
        height=height,
    )
    save_figure(
        fig, output_dir, filename, formats, _MARGINAL_BAR_WIDTH_PX, height
    )


def fig_topic_doc_count(
    topic_df: pd.DataFrame,
    output_dir: Path,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> None:
    _bar_chart(
        topic_df,
        label_col="label",
        value_col="doc_count",
        title="Document Count by Topic",
        xlabel="Documents",
        output_dir=output_dir,
        filename="fig_topic_doc_count",
        formats=formats,
    )


def fig_topic_token_count(
    topic_df: pd.DataFrame,
    output_dir: Path,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> None:
    _bar_chart(
        topic_df,
        label_col="label",
        value_col="token_count_est",
        title="Estimated Token Count by Topic",
        xlabel="Tokens (estimated)",
        output_dir=output_dir,
        filename="fig_topic_token_count",
        formats=formats,
    )


def fig_format_doc_count(
    format_df: pd.DataFrame,
    output_dir: Path,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> None:
    _bar_chart(
        format_df,
        label_col="label",
        value_col="doc_count",
        title="Document Count by Format",
        xlabel="Documents",
        output_dir=output_dir,
        filename="fig_format_doc_count",
        formats=formats,
    )


def fig_format_token_count(
    format_df: pd.DataFrame,
    output_dir: Path,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> None:
    _bar_chart(
        format_df,
        label_col="label",
        value_col="token_count_est",
        title="Estimated Token Count by Format",
        xlabel="Tokens (estimated)",
        output_dir=output_dir,
        filename="fig_format_token_count",
        formats=formats,
    )


__all__ = [
    "fig_format_doc_count",
    "fig_format_token_count",
    "fig_topic_doc_count",
    "fig_topic_token_count",
]
