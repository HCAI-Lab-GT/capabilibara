"""Paper-figure style system.

Centralized "Word-styles" registry for every figure in the manuscript.
Roles (TITLE, AXIS_TITLE, TICK, etc.) are defined once here; each
generator pulls fonts via ``plotly_font(role)`` or ``mpl_font(role)``
adapters. Changing a single role propagates to every figure on next
regeneration.

Backward-compatible shims (``FONT_SIZE``, ``TITLE_FONT_SIZE``,
``AXIS_TITLE_FONT_SIZE``) remain exported so older callers do not break
mid-refactor, but new code should prefer ``style_size("TICK")``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Role-based style registry — single source of truth for every figure font.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TextStyle:
    family: str = "Times New Roman"
    size: int = 10
    weight: str = "normal"   # "normal" | "bold"
    color: str = "#1a1a1a"


# One row per visual role. Tune these to retune every figure in the paper.
STYLES: dict[str, TextStyle] = {
    "TITLE":          TextStyle(size=14, weight="bold"),
    "SUBPLOT_TITLE":  TextStyle(size=11, weight="bold"),
    "AXIS_TITLE":     TextStyle(size=11),
    "TICK":           TextStyle(size=10),
    "COLORBAR_TICK":  TextStyle(size=7),
    "LEGEND":         TextStyle(size=10),
    "ANNOTATION":     TextStyle(size=7),
    "PANEL_LABEL":    TextStyle(size=11, weight="bold"),
    "CAPTION_HINT":   TextStyle(size=8),
}


def style(role: str) -> TextStyle:
    """Return the TextStyle for a role; raise KeyError on unknown role."""
    return STYLES[role]


def style_size(role: str) -> int:
    return STYLES[role].size


# Plotly fonts are measured in pixels of the *rendered* canvas. Our
# Plotly figures render at ~1400-2000 px wide and get scaled to
# \linewidth (~3.3 in / ~240 pt) by LaTeX, so each role-pt of intended
# displayed text needs ~2.5 px of Plotly font. Tune this single
# constant if Plotly figures look uniformly too big or too small after
# LaTeX scaling.
PLOTLY_SIZE_SCALE = 2.5


def plotly_font(role: str, scale: float = 1.0) -> dict:
    """Adapter: return a Plotly font dict for the named role.

    The role's intended pt size is multiplied by
    ``PLOTLY_SIZE_SCALE * scale`` so the figure reads at the right
    size after LaTeX scales the rendered PDF.

    ``scale`` lets callers compensate for figure-width: wide
    ``figure*`` (column-spanning) figures get less LaTeX downscaling,
    so callers should pass ``scale=0.6`` to shrink font sizes back to
    parity with non-wide figures. Single-column figures keep the
    default ``scale=1.0``.
    """
    s = STYLES[role]
    return {
        "family": s.family,
        "size": int(round(s.size * PLOTLY_SIZE_SCALE * scale)),
        "color": s.color,
    }


def mpl_font(role: str) -> dict:
    """Adapter: return a matplotlib fontdict for the named role.

    Matplotlib sizes are in pt; no scaling needed because matplotlib
    figures are sized in inches (typically already paper-scale).
    """
    s = STYLES[role]
    return {"family": s.family, "size": s.size, "weight": s.weight, "color": s.color}


# ---------------------------------------------------------------------------
# Backward-compatibility shims (referenced by older generator code).
# Prefer STYLES["..."] / style_size("...") in new code.
# ---------------------------------------------------------------------------

FONT_SIZE = STYLES["TICK"].size                       # 8
AXIS_TITLE_FONT_SIZE = STYLES["AXIS_TITLE"].size      # 10
TITLE_FONT_SIZE = STYLES["TITLE"].size                # 14

DPI_SCALE = 3
INFLUENCE_SCALE_FACTOR = 1.0
INFLUENCE_UNIT = "z-score"

FIGURE_WIDTH_PX = 900
FIGURE_HEIGHT_BAR_PX = 650
FIGURE_HEIGHT_HEATMAP_PX = 1100
FIGURE_WIDTH_HEATMAP_PX = 1400
FIGURE_HEIGHT_SIDE_BY_SIDE_PX = 1100
FIGURE_WIDTH_SIDE_BY_SIDE_PX = 1900

COLOR_PRIMARY = "#2c7bb6"
COLOR_SECONDARY = "#d7191c"
COLOR_EMPTY_CELL = "#f0f0f0"
COLOR_DIVERGING_POS = "#4575b4"
COLOR_DIVERGING_NEG = "#d73027"

DIVERGING_COLORSCALE = [
    # Negative = red, zero = white, positive = blue. Endpoint bands are
    # intentionally broad enough that saturated cells and the colorbar visibly
    # share the same dark red/blue colors after LaTeX scales the figure.
    [0.00, "#67001f"],
    [0.14, "#67001f"],
    [0.28, "#b2182b"],
    [0.42, "#ef8a62"],
    [0.50, "#f7f7f7"],
    [0.58, "#67a9cf"],
    [0.72, "#2166ac"],
    [0.86, "#053061"],
    [1.00, "#053061"],
]
PURPLES_COLORSCALE = "Purples"
COMPARISON_COLORSCALE = [
    [0.0, "#1b7837"],
    [0.5, "#f7f7f7"],
    [1.0, "#762a83"],
]


# ---------------------------------------------------------------------------
# Tick / colorbar helpers.
# ---------------------------------------------------------------------------

# Threshold below which diverging_colorbar_ticks switches to scientific
# notation. 0.01 catches the raw-influence range (~1e-5) without
# disturbing well-scaled z-score ranges (~1-3).
_SMALL_VALUE_THRESHOLD = 1e-2


def _fmt(value: float, decimals: int, use_scientific: bool) -> str:
    if use_scientific:
        return f"{value:.{decimals}e}"
    return f"{value:.{decimals}f}"


def linear_colorbar_ticks(
    vmin: float,
    vmax: float,
    n_ticks: int = 5,
    decimals: int = 2,
) -> tuple[list[float], list[str]]:
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == vmin:
        val = float(vmin) if np.isfinite(vmin) else 0.0
        return [val], [f"{val:.{decimals}f}"]
    n_ticks = max(int(n_ticks), 2)
    tick_vals = np.linspace(float(vmin), float(vmax), n_ticks).tolist()
    use_sci = max(abs(float(vmin)), abs(float(vmax))) < _SMALL_VALUE_THRESHOLD
    tick_text = [_fmt(v, decimals, use_sci) for v in tick_vals]
    return tick_vals, tick_text


def diverging_colorbar_ticks(
    abs_max: float,
    n_ticks: int = 5,
    decimals: int = 2,
) -> tuple[list[float], list[str]]:
    """Symmetric tick generator with auto sci-notation for tiny ranges.

    When ``abs_max`` is below ``_SMALL_VALUE_THRESHOLD`` (1e-2), the
    text uses scientific notation so raw-unit colorbars (~1e-5
    magnitude) stay readable instead of collapsing to "0.00".
    """
    if not np.isfinite(abs_max) or abs_max <= 0:
        return [0.0], [f"{0.0:.{decimals}f}"]
    n_ticks = max(int(n_ticks), 3)
    if n_ticks % 2 == 0:
        n_ticks += 1
    tick_vals = (np.linspace(-1.0, 1.0, n_ticks) * float(abs_max)).tolist()
    use_sci = abs(float(abs_max)) < _SMALL_VALUE_THRESHOLD
    tick_text = [
        "0" if abs(float(v)) < 1e-12 else _fmt(v, decimals, use_sci)
        for v in tick_vals
    ]
    return tick_vals, tick_text


def diverging_colorbar_strip(
    abs_max: float,
    *,
    n_ticks: int = 7,
    decimals: int = 1,
    n_steps: int = 256,
) -> tuple[np.ndarray, list[float], list[str]]:
    """Return a rasterized colorbar strip and matching tick labels.

    Plotly's native vector colorbar gradient renders differently in Apple's
    PDF stack than in Poppler. Using a one-column heatmap for the legend keeps
    the colorbar on the same rendering path as the data cells.
    """
    abs_max = float(abs_max)
    if not np.isfinite(abs_max) or abs_max <= 0:
        abs_max = 1.0
    values = np.linspace(-abs_max, abs_max, max(int(n_steps), 2))
    tick_vals, tick_text = diverging_colorbar_ticks(
        abs_max,
        n_ticks=n_ticks,
        decimals=decimals,
    )
    return values.reshape(-1, 1), tick_vals, tick_text


def sequential_colorbar_strip(
    vmin: float,
    vmax: float,
    *,
    tick_vals: list[float] | None = None,
    tick_text: list[str] | None = None,
    n_ticks: int = 5,
    decimals: int = 2,
    n_steps: int = 256,
) -> tuple[np.ndarray, list[float], list[str]]:
    """Return a one-column sequential colorbar strip and matching ticks."""
    vmin = float(vmin)
    vmax = float(vmax)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0
    if vmax < vmin:
        vmin, vmax = vmax, vmin
    if vmax == vmin:
        vmax = vmin + 1.0
    values = np.linspace(vmin, vmax, max(int(n_steps), 2))
    if tick_vals is None or tick_text is None:
        tick_vals, tick_text = linear_colorbar_ticks(
            vmin,
            vmax,
            n_ticks=n_ticks,
            decimals=decimals,
        )
    return values.reshape(-1, 1), tick_vals, tick_text


def apply_colorbar_strip_axes(
    fig,
    *,
    row: int,
    col: int,
    tick_vals: list[float],
    tick_text: list[str],
    title: str,
    title_x: float,
    title_y: float,
    tick_scale: float = 0.8,
    title_scale: float = 0.9,
) -> None:
    """Style a manual one-column heatmap strip as a colorbar."""
    fig.update_xaxes(
        showticklabels=False,
        ticks="",
        showline=False,
        row=row,
        col=col,
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_text,
        tickfont=plotly_font("COLORBAR_TICK", scale=tick_scale),
        ticks="outside",
        ticklen=4,
        tickwidth=1,
        tickcolor="#222222",
        side="right",
        showline=False,
        row=row,
        col=col,
    )
    fig.add_annotation(
        text=title,
        x=title_x,
        y=title_y,
        xref="paper",
        yref="paper",
        showarrow=False,
        textangle=-90,
        font=plotly_font("COLORBAR_TICK", scale=title_scale),
    )


def symmetric_color_limit(
    values,
    *,
    percentile: float = 95.0,
    min_limit: float = 1e-8,
) -> float:
    """Return a robust symmetric color limit for diverging heatmaps.

    Influence z-scores can contain a handful of extreme cells. Using the
    full min/max makes the rest of the heatmap visually collapse toward
    white; a percentile limit preserves readable structure while keeping
    a single symmetric scale across panels.
    """
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float(min_limit)
    limit = float(np.nanpercentile(np.abs(finite), percentile))
    if not np.isfinite(limit):
        return float(min_limit)
    return float(max(limit, min_limit))


def diverging_coloraxis(
    abs_max: float,
    title: str,
    *,
    n_ticks: int = 7,
    decimals: int = 2,
    length: float = 0.96,
    thickness: int = 22,
    y: float = 0.5,
) -> dict:
    """Shared Plotly diverging coloraxis for signed heatmaps.

    The manuscript convention is negative = red, zero = white,
    positive = blue. Keeping this as one helper prevents the colorbar
    drift that previously made some exported legends look one-sided.
    """
    abs_max = float(abs_max)
    if not np.isfinite(abs_max) or abs_max <= 0:
        abs_max = 1.0
    tick_vals, tick_text = diverging_colorbar_ticks(
        abs_max,
        n_ticks=n_ticks,
        decimals=decimals,
    )
    return {
        "colorscale": DIVERGING_COLORSCALE,
        "cmin": -abs_max,
        "cmax": abs_max,
        "cmid": 0,
        "colorbar": {
            "title": {"text": title, "side": "right"},
            "tickmode": "array",
            "tickvals": tick_vals,
            "ticktext": tick_text,
            "tickfont": plotly_font("COLORBAR_TICK"),
            "ticks": "outside",
            "ticklen": 4,
            "tickwidth": 1,
            "len": length,
            "y": y,
            "yanchor": "middle",
            "thickness": thickness,
            "outlinewidth": 0,
        },
    }


def apply_heatmap_axes(
    fig,
    *,
    x_labels: list[str] | None = None,
    y_labels: list[str] | None = None,
    tick_scale: float = 0.6,
    x_angle: int = 45,
) -> None:
    """Apply the manuscript heatmap axis style to every subplot axis."""
    axis_common = {
        "ticks": "outside",
        "ticklen": 4,
        "tickwidth": 1,
        "tickcolor": "#222222",
        "showline": True,
        "linecolor": "#222222",
        "linewidth": 1,
        "mirror": False,
        "showgrid": False,
        "zeroline": False,
    }
    xaxis: dict[str, object] = dict(
        **axis_common,
        tickangle=x_angle,
        tickfont=plotly_font("TICK", scale=tick_scale),
    )
    yaxis: dict[str, object] = dict(
        **axis_common,
        tickfont=plotly_font("TICK", scale=tick_scale),
    )
    if x_labels is not None:
        xaxis.update(tickmode="array", tickvals=x_labels, ticktext=x_labels)
    if y_labels is not None:
        yaxis.update(tickmode="array", tickvals=y_labels, ticktext=y_labels)
    fig.update_xaxes(**xaxis)
    fig.update_yaxes(**yaxis)


# ---------------------------------------------------------------------------
# Plotly layout helpers.
# ---------------------------------------------------------------------------

def paper_layout(title: str | None = None, **kwargs) -> dict:
    """Return a Plotly layout dict pre-populated with paper-quality defaults.

    Pass ``title=None`` (default) to suppress the in-image title — preferred
    for figures whose title is carried by the LaTeX caption.
    """
    base = {
        "font": plotly_font("TICK"),  # default body text is tick-size
        "plot_bgcolor": "white",
        "paper_bgcolor": "white",
        "margin": {"l": 140, "r": 40, "t": 80, "b": 120},
    }
    if title:
        base["title"] = {
            "text": title,
            "x": 0.5,
            "xanchor": "center",
            "font": plotly_font("TITLE"),
        }
    else:
        # Explicit empty title so Plotly doesn't auto-add one
        base["title"] = {"text": ""}
        base["margin"]["t"] = 30
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# Matplotlib helpers.
# ---------------------------------------------------------------------------

def mpl_apply_axes(ax) -> None:
    """Apply unified tick + axis-title font roles to a matplotlib Axes.

    Call after plotting; safe to call repeatedly.
    """
    tick_font = mpl_font("TICK")
    axis_font = mpl_font("AXIS_TITLE")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(tick_font["size"])
        label.set_fontfamily(tick_font["family"])
        label.set_color(tick_font["color"])
    if ax.get_xlabel():
        ax.set_xlabel(ax.get_xlabel(), **axis_font)
    if ax.get_ylabel():
        ax.set_ylabel(ax.get_ylabel(), **axis_font)
    if ax.get_title():
        ax.set_title(ax.get_title(), **mpl_font("SUBPLOT_TITLE"))


def save_figure(
    fig,
    output_dir: Path,
    filename: str,
    formats: tuple[str, ...],
    width: int = FIGURE_WIDTH_PX,
    height: int = FIGURE_HEIGHT_BAR_PX,
) -> None:
    output_dir = Path(output_dir)
    for fmt in formats:
        path = output_dir / f"{filename}.{fmt}"
        if fmt == "html":
            fig.write_html(str(path), include_plotlyjs="cdn")
        elif fmt == "pdf":
            fig.write_image(str(path), width=width, height=height)
        else:
            fig.write_image(str(path), width=width, height=height, scale=DPI_SCALE)
        logger.debug("Saved %s", path)


__all__ = [
    # Role-based style API (preferred)
    "STYLES",
    "TextStyle",
    "style",
    "style_size",
    "plotly_font",
    "mpl_font",
    "mpl_apply_axes",
    # Layout + helpers
    "paper_layout",
    "linear_colorbar_ticks",
    "diverging_colorbar_ticks",
    "diverging_colorbar_strip",
    "sequential_colorbar_strip",
    "apply_colorbar_strip_axes",
    "diverging_coloraxis",
    "symmetric_color_limit",
    "apply_heatmap_axes",
    "save_figure",
    # Backwards-compat constants
    "AXIS_TITLE_FONT_SIZE",
    "FONT_SIZE",
    "TITLE_FONT_SIZE",
    "DPI_SCALE",
    "INFLUENCE_SCALE_FACTOR",
    "INFLUENCE_UNIT",
    # Colors + colorscales
    "COLOR_DIVERGING_NEG",
    "COLOR_DIVERGING_POS",
    "COLOR_EMPTY_CELL",
    "COLOR_PRIMARY",
    "COLOR_SECONDARY",
    "COMPARISON_COLORSCALE",
    "DIVERGING_COLORSCALE",
    "PURPLES_COLORSCALE",
    # Sizing
    "FIGURE_HEIGHT_BAR_PX",
    "FIGURE_HEIGHT_HEATMAP_PX",
    "FIGURE_HEIGHT_SIDE_BY_SIDE_PX",
    "FIGURE_WIDTH_HEATMAP_PX",
    "FIGURE_WIDTH_PX",
    "FIGURE_WIDTH_SIDE_BY_SIDE_PX",
]
