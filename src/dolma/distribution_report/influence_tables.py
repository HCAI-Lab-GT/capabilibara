"""Top-bin ranking and correctness stratification tables."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from dolma.distribution_report.data_loader import display_label
from dolma.distribution_report.influence_loader import BENCHMARK_DISPLAY_NAMES

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 20


_LATEX_SPECIAL_CHARS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _latex_escape(text: str) -> str:
    return "".join(_LATEX_SPECIAL_CHARS.get(ch, ch) for ch in text)


def _fmt_float_latex(value: float, decimals: int = 2) -> str:
    if pd.isna(value):
        return ""
    rendered = f"{abs(value):.{decimals}f}"
    if value < 0:
        return f"$-${rendered}"
    return rendered


def write_contrastive_bins_latex(
    table_df: pd.DataFrame,
    key_a: str,
    key_b: str,
    output_dir: Path,
    *,
    decimals: int = 2,
) -> Path:
    """Write a LaTeX table for the contrastive bins output.

    This is intended for paper inclusion. Values are assumed to already be
    z-score standardized (per-benchmark) upstream.
    """

    short_a = key_a.removeprefix("queries_")
    short_b = key_b.removeprefix("queries_")

    display_a = _latex_escape(BENCHMARK_DISPLAY_NAMES.get(key_a, short_a))
    display_b = _latex_escape(BENCHMARK_DISPLAY_NAMES.get(key_b, short_b))

    delta_col = r"$\Delta$"

    df = table_df.reset_index().rename(
        columns={
            "rank": "Rank",
            "bin_label": "Topic / Format",
            f"mean_score_{short_a}": display_a,
            f"mean_score_{short_b}": display_b,
            "diff": delta_col,
        }
    )

    df["Topic / Format"] = df["Topic / Format"].map(_latex_escape)
    for col in (display_a, display_b, delta_col):
        df[col] = df[col].map(lambda x: _fmt_float_latex(x, decimals=decimals))

    lines: list[str] = []
    lines.append(r"\begin{tabular}{rlrrr}")
    lines.append(r"\toprule")
    lines.append(
        "Rank & Topic / Format & " + f"{display_a} & {display_b} & $\\Delta$ \\\\"
    )
    lines.append(
        r"     &                & \multicolumn{3}{c}{(mean z-score; $\Delta$ = z-score difference)} \\\\"
    )
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        lines.append(
            f"{int(row['Rank']):2d} & {row['Topic / Format']} & {row[display_a]} & {row[display_b]} & {row[delta_col]} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    latex = "\n".join(lines) + "\n"

    out_path = output_dir / f"table_contrastive_{short_a}_vs_{short_b}.tex"
    out_path.write_text(latex)
    logger.info("Wrote %s", out_path)
    return out_path


def _bin_label(topic: str, fmt: str) -> str:
    return f"{display_label(topic)} / {display_label(fmt)}"


def top_bins_table(
    grid_df: pd.DataFrame,
    benchmark_key: str,
    output_dir: Path,
    top_k: int = DEFAULT_TOP_K,
) -> pd.DataFrame:
    short = benchmark_key.removeprefix("queries_")
    df = grid_df.copy()
    df["bin_label"] = df.apply(
        lambda r: _bin_label(r["topic_label"], r["format_label"]), axis=1
    )

    signed = df.nlargest(top_k, "mean_score")[
        ["bin_label", "mean_score", "abs_mean_score", "doc_count"]
    ].reset_index(drop=True)
    signed.index = signed.index + 1
    signed.index.name = "rank"

    out_path = output_dir / f"table_top_bins_{short}.csv"
    signed.to_csv(out_path)
    logger.info("Wrote %s (%d rows)", out_path, len(signed))
    return signed


def contrastive_bins_table(
    grid_a: pd.DataFrame,
    grid_b: pd.DataFrame,
    key_a: str,
    key_b: str,
    output_dir: Path,
    top_k: int = DEFAULT_TOP_K,
) -> pd.DataFrame:
    short_a = key_a.removeprefix("queries_")
    short_b = key_b.removeprefix("queries_")

    merged = (
        grid_a[["topic_label", "format_label", "mean_score"]]
        .merge(
            grid_b[["topic_label", "format_label", "mean_score"]],
            on=["topic_label", "format_label"],
            suffixes=(f"_{short_a}", f"_{short_b}"),
            how="outer",
        )
        .fillna(0)
    )
    merged["diff"] = merged[f"mean_score_{short_a}"] - merged[f"mean_score_{short_b}"]
    merged["bin_label"] = merged.apply(
        lambda r: _bin_label(r["topic_label"], r["format_label"]), axis=1
    )

    favor_a = merged.nlargest(top_k, "diff")[
        ["bin_label", f"mean_score_{short_a}", f"mean_score_{short_b}", "diff"]
    ].reset_index(drop=True)
    favor_a.index = favor_a.index + 1
    favor_a.index.name = "rank"

    out_path = output_dir / f"table_contrastive_{short_a}_vs_{short_b}.csv"
    favor_a.to_csv(out_path)
    logger.info("Wrote %s (%d rows)", out_path, len(favor_a))
    return favor_a


def correctness_diff_table(
    correct_grid: pd.DataFrame,
    incorrect_grid: pd.DataFrame,
    benchmark_key: str,
    output_dir: Path,
    top_k: int = DEFAULT_TOP_K,
) -> pd.DataFrame:
    short = benchmark_key.removeprefix("queries_")
    merged = (
        correct_grid[["topic_label", "format_label", "mean_score"]]
        .merge(
            incorrect_grid[["topic_label", "format_label", "mean_score"]],
            on=["topic_label", "format_label"],
            suffixes=("_correct", "_incorrect"),
            how="outer",
        )
        .fillna(0)
    )
    merged["diff"] = merged["mean_score_correct"] - merged["mean_score_incorrect"]
    merged["bin_label"] = merged.apply(
        lambda r: _bin_label(r["topic_label"], r["format_label"]), axis=1
    )

    by_abs_diff = (
        merged.reindex(merged["diff"].abs().sort_values(ascending=False).index)
        .head(top_k)[
            ["bin_label", "mean_score_correct", "mean_score_incorrect", "diff"]
        ]
        .reset_index(drop=True)
    )
    by_abs_diff.index = by_abs_diff.index + 1
    by_abs_diff.index.name = "rank"

    out_path = output_dir / f"table_correctness_diff_{short}.csv"
    by_abs_diff.to_csv(out_path)
    logger.info("Wrote %s (%d rows)", out_path, len(by_abs_diff))
    return by_abs_diff


__all__ = [
    "contrastive_bins_table",
    "correctness_diff_table",
    "top_bins_table",
    "write_contrastive_bins_latex",
]
