"""Composite signed topic and format influence bar figures (6-panel, 3x2)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dolma.distribution_report.influence_composite_6way import _PANEL_ORDER_6WAY
from dolma.distribution_report.influence_composite_signed_topic import (
    FIGURE_HEIGHT_4BAR_PX,
    _signed_marginals,
    _signed_panel_figure,
)

FIGURE_WIDTH_6BAR_PX = 3200
FIGURE_HEIGHT_6BAR_PX = FIGURE_HEIGHT_4BAR_PX


def fig_influence_signed_topic_6panel(
    grids: dict[str, pd.DataFrame],
    output_dir: Path,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> None:
    marginals = _signed_marginals(grids, list(_PANEL_ORDER_6WAY), "topic_label")
    _signed_panel_figure(
        marginals,
        list(_PANEL_ORDER_6WAY),
        axis_label="Topic",
        title="Mean Signed Influence by Topic",
        filename="fig_influence_signed_topic_6panel",
        output_dir=output_dir,
        formats=formats,
        nrows=2,
        ncols=3,
        width=FIGURE_WIDTH_6BAR_PX,
        height=FIGURE_HEIGHT_6BAR_PX,
    )


def fig_influence_signed_format_6panel(
    grids: dict[str, pd.DataFrame],
    output_dir: Path,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> None:
    marginals = _signed_marginals(grids, list(_PANEL_ORDER_6WAY), "format_label")
    _signed_panel_figure(
        marginals,
        list(_PANEL_ORDER_6WAY),
        axis_label="Format",
        title="Mean Signed Influence by Format",
        filename="fig_influence_signed_format_6panel",
        output_dir=output_dir,
        formats=formats,
        nrows=2,
        ncols=3,
        width=FIGURE_WIDTH_6BAR_PX,
        height=FIGURE_HEIGHT_6BAR_PX,
    )


__all__ = [
    "FIGURE_HEIGHT_6BAR_PX",
    "FIGURE_WIDTH_6BAR_PX",
    "fig_influence_signed_format_6panel",
    "fig_influence_signed_topic_6panel",
]
