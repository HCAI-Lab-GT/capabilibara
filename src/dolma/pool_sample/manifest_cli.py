"""CLI for drawing sampling manifests from a SOC-95-style corpus manifest."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from dolma.constants import TARGET_DOCS_PER_BIN
from dolma.distribution_report.sampling_loader import load_sampleable_manifest
from dolma.pool_sample.sampling import (
    REPRESENTATIVE_TOKEN_TARGET,
    compute_bin_stats,
    generate_dummy_manifest,
    representative_sample,
    stratified_sample,
)

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw stratified and representative samples from a corpus manifest."
    )
    parser.add_argument(
        "--dummy", action="store_true", help="Use dummy data for testing."
    )
    parser.add_argument(
        "--input", type=Path, default=None, help="Path to source manifest."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/manifests"),
        help="Directory for sampled manifests and bin stats.",
    )
    parser.add_argument(
        "--n-dummy-docs",
        type=int,
        default=500_000,
        help="Number of dummy docs to generate with --dummy.",
    )
    parser.add_argument(
        "--target-docs-per-bin",
        type=int,
        default=TARGET_DOCS_PER_BIN,
        help="Target docs per bin for the stratified sample.",
    )
    parser.add_argument(
        "--representative-token-target",
        type=int,
        default=REPRESENTATIVE_TOKEN_TARGET,
        help="Token budget for the representative sample.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stratified_path = output_dir / "pool_stratified_manifest.parquet"
    representative_path = output_dir / "pool_representative_manifest.parquet"
    bin_stats_path = output_dir / "pool_bin_stats.parquet"

    if args.dummy:
        logger.info("Generating dummy manifest with %d documents", args.n_dummy_docs)
        manifest = generate_dummy_manifest(n_docs=args.n_dummy_docs)
    elif args.input is not None:
        logger.info("Loading manifest from %s", args.input)
        manifest = load_sampleable_manifest(args.input)
    else:
        raise ValueError("Either --dummy or --input must be specified")

    bin_stats = compute_bin_stats(
        manifest, target_docs_per_bin=args.target_docs_per_bin
    )
    bin_stats.to_parquet(bin_stats_path, index=False)
    logger.info("Wrote %s", bin_stats_path)

    stratified_df = stratified_sample(
        manifest,
        bin_stats,
        target_docs_per_bin=args.target_docs_per_bin,
    )
    stratified_df.to_parquet(stratified_path, index=False)
    logger.info("Wrote %s", stratified_path)

    representative_df = representative_sample(
        manifest,
        stratified_df,
        token_target=args.representative_token_target,
    )
    representative_df.to_parquet(representative_path, index=False)
    logger.info("Wrote %s", representative_path)

    logger.info(
        "Sampling complete: %d stratified docs, %d representative docs",
        len(stratified_df),
        len(representative_df),
    )
    return 0


__all__ = ["main", "parse_args"]


if __name__ == "__main__":
    raise SystemExit(main())
