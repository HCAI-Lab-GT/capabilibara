"""Command-line export for SocialIQA queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections.abc import Mapping, Sequence

import pyarrow as pa
from pyarrow import ipc

from data_attribution.data.socialiqa import SocialIQAManifest, socialiqa_manifest


def _write_jsonl(records: Sequence[Mapping[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            json.dump(record, stream)
            stream.write("\n")


def _write_arrow(
    records: Sequence[Mapping[str, object]], metadata: Mapping[str, object], path: Path
) -> None:
    table = pa.Table.from_pylist(list(records))
    schema_metadata = table.schema.metadata or {}
    encoded_metadata = {
        key: str(value).encode("utf-8") for key, value in metadata.items()
    }
    table = table.replace_schema_metadata({**schema_metadata, **encoded_metadata})

    path.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(path, "wb") as sink:
        ipc.write_table(table, sink)


def _write_metadata_file(metadata: Mapping[str, object], output_path: Path) -> None:
    metadata_path = output_path.with_suffix(f"{output_path.suffix}.metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")


def _metadata_from_manifest(manifest: SocialIQAManifest) -> dict[str, object]:
    return {
        "dataset": manifest.dataset,
        "split": manifest.split,
        "seed": manifest.seed,
        "sample_size": manifest.sample_size,
        "query_count": manifest.query_count,
    }


def export_socialiqa_queries(
    *,
    split: str,
    sample_size: int | None,
    seed: int,
    output: Path,
) -> None:
    manifest = socialiqa_manifest(split=split, sample_size=sample_size, seed=seed)
    metadata = _metadata_from_manifest(manifest)

    if output.suffix.lower() == ".jsonl":
        _write_jsonl(manifest.queries, output)
    elif output.suffix.lower() == ".arrow":
        _write_arrow(manifest.queries, metadata, output)
    else:
        raise ValueError(
            f"Unsupported output format for {output}; use .jsonl or .arrow"
        )

    _write_metadata_file(metadata, output)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export SocialIQA queries to JSONL or Arrow"
    )
    parser.add_argument("--split", default="validation", help="Dataset split to export")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional number of examples to sample",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    parser.add_argument(
        "--output", required=True, type=Path, help="Destination .jsonl or .arrow file"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    export_socialiqa_queries(
        split=args.split,
        sample_size=args.sample_size,
        seed=args.seed,
        output=args.output,
    )


if __name__ == "__main__":
    main()
