"""CLI entry point for WebOrganizer report generation."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from dolma.distribution_report.runner import run_report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reusable WebOrganizer report figures and tables."
    )
    parser.add_argument("--eda-dir", type=Path, default=Path("artifacts/dolma_eda"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/paper_figures")
    )
    parser.add_argument(
        "--run-label",
        help=(
            "Human-readable label recorded in report outputs. Defaults to the "
            "output directory name."
        ),
    )
    parser.add_argument(
        "--format",
        choices=["pdf", "png", "html", "both", "all"],
        default="both",
        dest="output_format",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Generate sampling comparison figures with dummy data.",
    )
    parser.add_argument(
        "--representative-manifest",
        type=Path,
        help=(
            "Local parquet path for the representative sample manifest. "
            "Canonical remote source: HCAI-Lab/archive-dolma3-pool-150b-samples."
        ),
    )
    parser.add_argument(
        "--stratified-manifest",
        type=Path,
        help=(
            "Local parquet path for the stratified sample manifest. "
            "Canonical remote source: HCAI-Lab/archive-dolma3-pool-150b-samples."
        ),
    )
    parser.add_argument(
        "--influence-dir",
        type=Path,
        help="Directory with bin-level influence CSVs from trackstar-bin-aggregate.",
    )
    parser.add_argument(
        "--influence-split-dir",
        type=Path,
        help="Directory with correct/incorrect split CSVs from trackstar-bin-aggregate-split.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return run_report(
        eda_dir=args.eda_dir,
        output_dir=args.output_dir,
        output_format=args.output_format,
        run_label=args.run_label,
        use_dummy=args.dummy,
        representative_manifest=args.representative_manifest,
        stratified_manifest=args.stratified_manifest,
        influence_dir=args.influence_dir,
        influence_split_dir=args.influence_split_dir,
    )


__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
