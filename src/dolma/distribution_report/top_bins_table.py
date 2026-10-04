"""Top-N and bottom-N bin tables ranked by token mass for paper appendix."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dolma.distribution_report.data_loader import display_label


def _latex_safe(text: str) -> str:
    """Escape LaTeX special characters that appear in our label vocabulary.

    Without this, a label like "Q&A Forum" inserts a literal `&` into a
    tabular row, which LaTeX interprets as a column separator — splitting
    "Q&A Forum" across two cells and shoving downstream columns onto a
    new visual line. Only `&` actually appears in our short/display label
    sets today; `%`, `_`, `$`, and `#` are added defensively.
    """
    return (
        text.replace("\\", r"\textbackslash{}")
            .replace("&", r"\&")
            .replace("%", r"\%")
            .replace("_", r"\_")
            .replace("$", r"\$")
            .replace("#", r"\#")
    )


def _humanize_count(n: float) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 10_000:
        return f"{n / 1_000:.0f}K"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return f"{int(round(n))}"


def build_ranked(grid_df: pd.DataFrame, n: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (top_n, bottom_n) dataframes ranked by token mass.

    Bottom_n is ordered ascending (smallest bin first) so the worst-populated
    bin is at the top of the bottom table.
    """
    required = {"topic_label", "format_label", "doc_count", "token_count_est"}
    missing = required - set(grid_df.columns)
    if missing:
        raise ValueError(f"grid_df missing required columns: {sorted(missing)}")

    total_tokens = float(grid_df["token_count_est"].sum())
    df = grid_df.copy()
    df["tokens_b"] = df["token_count_est"] / 1e9
    df["share_pct"] = df["token_count_est"] / total_tokens * 100.0
    df = df.sort_values("token_count_est", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    top = df.head(n).copy()
    # Round 8: keep DESCENDING (rank 557 at top, rank 576 at bottom) so
    # the worst-ranked bin lands at the bottom of the panel — matches
    # the natural reading order user expects ("descending rank so the
    # worst rank should be at the bottom, not the top").
    bottom = df.tail(n).copy()
    return top, bottom


def to_latex_tabular(df: pd.DataFrame) -> str:
    """Render a ranked sub-frame as a standalone LaTeX tabular block."""
    rows = []
    for _, row in df.iterrows():
        topic = _latex_safe(display_label(row["topic_label"]))
        fmt = _latex_safe(display_label(row["format_label"]))
        docs = _humanize_count(float(row["doc_count"]))
        rows.append(
            f"{int(row['rank'])} & {topic} & {fmt} & "
            f"{docs} & {row['tokens_b']:.2f} & {row['share_pct']:.2f} \\\\"
        )
    body = "\n".join(rows)
    return (
        "\\begin{tabular}{rllrrr}\n"
        "\\toprule\n"
        "\\textbf{Rank} & \\textbf{Topic} & \\textbf{Format} & "
        "\\textbf{Docs} & \\textbf{Tokens (B)} & \\textbf{Share (\\%)} \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )


def write_outputs(
    grid_df: pd.DataFrame,
    output_dir: Path,
    n: int = 20,
) -> dict[str, Path]:
    """Write CSV + top/bottom LaTeX tabular leaves. Returns paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    top, bottom = build_ranked(grid_df, n=n)

    top_out = top.assign(side="top")[
        ["rank", "topic_label", "format_label", "doc_count", "tokens_b", "share_pct", "side"]
    ]
    bottom_out = bottom.assign(side="bottom")[
        ["rank", "topic_label", "format_label", "doc_count", "tokens_b", "share_pct", "side"]
    ]
    combined = pd.concat([top_out, bottom_out], ignore_index=True)

    csv_path = output_dir / "top_bottom_bins.csv"
    combined.to_csv(csv_path, index=False)

    top_tex = output_dir / "tab-top-bins.tabular-top.tex"
    bottom_tex = output_dir / "tab-top-bins.tabular-bottom.tex"
    top_tex.write_text(to_latex_tabular(top))
    bottom_tex.write_text(to_latex_tabular(bottom))

    return {"csv": csv_path, "top_tex": top_tex, "bottom_tex": bottom_tex}


__all__ = ["build_ranked", "to_latex_tabular", "write_outputs"]
