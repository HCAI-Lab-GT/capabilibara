#!/usr/bin/env python3
"""Export per-bin influence scores with z-scores added.

Reads aggregated and correctness-split per-bin CSVs from artifacts/
and writes one z-scored CSV per benchmark to artifacts/zscored_bin_scores/.

Z-score = (mean - mean_over_bins) / std_over_bins, computed independently
per benchmark over its 576 (topic x format) bins. NaN/Inf inputs are
dropped before standardization; bins missing from the input remain absent
from the output.

Output schema: topic_label, format_label, mean_score, zscore, doc_count.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS = REPO_ROOT / "artifacts"
OUTPUT_ROOT = ARTIFACTS / "zscored_bin_scores"


@dataclass(frozen=True)
class Source:
    output_name: str
    input_path: Path
    mean_col: str


def aggregated_sources() -> list[Source]:
    plain_dir = ARTIFACTS / "influence_bin_scores"
    base_dir = ARTIFACTS / "influence_bbh_base"
    instr_dir = ARTIFACTS / "influence_bbh_instruct"

    sources: list[Source] = []

    for bench in ("socialiqa", "gsm8k", "mmlu_social_science", "mmlu_stem"):
        sources.append(Source(
            output_name=bench,
            input_path=plain_dir / f"queries_{bench}_bin_scores.csv",
            mean_col="mean_score",
        ))

    for bench in ("arc_easy", "arc_challenge"):
        sources.append(Source(
            output_name=bench,
            input_path=plain_dir / f"queries_{bench}_bin_scores_perquery.csv",
            mean_col="mean_influence",
        ))
    # NOTE: influence_bin_scores/queries_bbh_snarks_bin_scores_perquery.csv is
    # byte-identical to influence_bbh_instruct/queries_bbh_snarks_bin_scores_perquery.csv
    # (same MD5). Only the SOC-170 base/instruct variants are exported below.

    for task in ("snarks", "causal_judgement", "sports_understanding"):
        sources.append(Source(
            output_name=f"bbh_{task}_base",
            input_path=base_dir / f"queries_bbh_{task}_bin_scores_perquery.csv",
            mean_col="mean_influence",
        ))
        sources.append(Source(
            output_name=f"bbh_{task}_instruct",
            input_path=instr_dir / f"queries_bbh_{task}_bin_scores_perquery.csv",
            mean_col="mean_influence",
        ))

    return sources


def split_sources() -> list[Source]:
    split_dir = ARTIFACTS / "influence_bin_scores_split"
    base_split_dir = ARTIFACTS / "influence_bbh_base_split"
    instr_split_dir = ARTIFACTS / "influence_bbh_instruct_split"

    sources: list[Source] = []

    benches = ("socialiqa", "gsm8k", "mmlu_social_science", "mmlu_stem",
               "arc_easy", "arc_challenge")
    for bench in benches:
        for split in ("correct", "incorrect"):
            sources.append(Source(
                output_name=f"{bench}_{split}",
                input_path=split_dir / f"queries_{bench}_bin_scores_{split}.csv",
                mean_col="mean_score",
            ))

    for task in ("snarks", "causal_judgement", "sports_understanding"):
        for split in ("correct", "incorrect"):
            sources.append(Source(
                output_name=f"bbh_{task}_base_{split}",
                input_path=base_split_dir / f"soc170_queries_olmes_bbh_{task}_bin_scores_{split}.csv",
                mean_col="mean_score",
            ))
            sources.append(Source(
                output_name=f"bbh_{task}_instruct_{split}",
                input_path=instr_split_dir / f"soc170_queries_olmes_instruct_bbh_{task}_bin_scores_{split}.csv",
                mean_col="mean_score",
            ))

    return sources


def export(source: Source, output_dir: Path) -> tuple[str, int, float, float] | None:
    if not source.input_path.exists():
        print(f"  SKIP (missing): {source.input_path.relative_to(REPO_ROOT)}",
              file=sys.stderr)
        return None

    df = pd.read_csv(source.input_path)
    if source.mean_col not in df.columns:
        print(f"  SKIP ({source.mean_col} not in columns): "
              f"{source.input_path.relative_to(REPO_ROOT)}", file=sys.stderr)
        return None

    df = df[["topic_label", "format_label", source.mean_col, "doc_count"]].copy()
    df = df.rename(columns={source.mean_col: "mean_score"})
    df = df.dropna(subset=["mean_score"])

    mu = df["mean_score"].mean()
    sigma = df["mean_score"].std()
    if sigma == 0 or pd.isna(sigma):
        print(f"  SKIP (zero/NaN std): {source.output_name}", file=sys.stderr)
        return None

    df["zscore"] = (df["mean_score"] - mu) / sigma
    df = df[["topic_label", "format_label", "mean_score", "zscore", "doc_count"]]

    out_path = output_dir / f"zscored_{source.output_name}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return (source.output_name, len(df), float(mu), float(sigma))


def write_readme(output_dir: Path, agg_rows: list, split_rows: list) -> None:
    lines = [
        "# Z-scored per-bin influence scores",
        "",
        "Per-benchmark CSVs derived from `artifacts/influence_*`. Each CSV "
        "has 576 rows or fewer (one per WebOrganizer topic x format bin).",
        "",
        "## Schema",
        "",
        "| column | meaning |",
        "|--------|---------|",
        "| `topic_label` | WebOrganizer topic (24 values) |",
        "| `format_label` | WebOrganizer format (24 values) |",
        "| `mean_score` | Raw mean influence score for documents in this bin |",
        "| `zscore` | `(mean_score - mean_over_bins) / std_over_bins`, per benchmark |",
        "| `doc_count` | Number of source documents in this bin |",
        "",
        "Z-score is computed per file independently. Cross-file comparisons "
        "are valid on z-scores but not on raw `mean_score` (the raw scale "
        "differs by orders of magnitude across benchmarks).",
        "",
        "## Files",
        "",
        "### Aggregated (`aggregated/`)",
        "",
        "One CSV per benchmark, scored over the full query set.",
        "",
        "| file | rows | mu | sigma |",
        "|------|-----:|---:|------:|",
    ]
    for name, n, mu, sigma in agg_rows:
        lines.append(f"| `zscored_{name}.csv` | {n} | {mu:.6g} | {sigma:.6g} |")
    lines += [
        "",
        "### Correctness splits (`splits/`)",
        "",
        "Z-scoring is done separately for the correct- and incorrect-answer "
        "query subsets. `mu` and `sigma` below are the per-subset values "
        "used for that file's z-score.",
        "",
        "| file | rows | mu | sigma |",
        "|------|-----:|---:|------:|",
    ]
    for name, n, mu, sigma in split_rows:
        lines.append(f"| `zscored_{name}.csv` | {n} | {mu:.6g} | {sigma:.6g} |")
    lines += [
        "",
        "## Notes on BBH",
        "",
        "Two variants per task:",
        "- `bbh_<task>_base`: SOC-170 base-model BBH scoring "
        "  (`influence_bbh_base/`).",
        "- `bbh_<task>_instruct`: SOC-170 instruct-model BBH scoring "
        "  (`influence_bbh_instruct/`).",
        "",
        "The older `influence_bin_scores/queries_bbh_snarks_bin_scores_perquery.csv` "
        "is byte-identical to the SOC-170 instruct snarks output and is not exported "
        "separately.",
        "",
        "## Regeneration",
        "",
        "```bash",
        "uv run python scripts/analysis/export_zscored_bin_scores.py",
        "```",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    agg_out = OUTPUT_ROOT / "aggregated"
    split_out = OUTPUT_ROOT / "splits"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Writing z-scored CSVs to {OUTPUT_ROOT.relative_to(REPO_ROOT)}")

    print("Aggregated:")
    agg_rows = [r for r in (export(s, agg_out) for s in aggregated_sources())
                if r is not None]
    print("Splits:")
    split_rows = [r for r in (export(s, split_out) for s in split_sources())
                  if r is not None]

    write_readme(OUTPUT_ROOT, agg_rows, split_rows)
    print(f"Done. {len(agg_rows)} aggregated + {len(split_rows)} split CSVs.")


if __name__ == "__main__":
    main()
