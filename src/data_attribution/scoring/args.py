from __future__ import annotations

import argparse
from pathlib import Path
from collections.abc import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score queries against a Bergson training index."
    )
    parser.add_argument(
        "--training-index",
        type=Path,
        required=True,
        help="Path to the training index directory",
    )
    parser.add_argument(
        "--query-index",
        type=Path,
        required=True,
        help="Path to the query index directory",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Destination JSON path for results"
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="Number of results to keep per query"
    )
    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=128,
        help="Number of queries to score per batch. Trades speed for memory.",
    )
    parser.add_argument(
        "--gradient-key",
        default=None,
        help="Specific gradient key to score. Defaults to the intersection of training/query keys.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional torch device for scoring (e.g., cuda:0). Defaults to Attributor settings.",
    )
    parser.add_argument(
        "--doc-id-field",
        default="doc_id",
        help="Field in data.hf to use as training document identifier. Fallback is positional index.",
    )
    parser.add_argument(
        "--query-id-field",
        default="query_id",
        help="Field in data.hf to use as query identifier. Fallback is positional index.",
    )
    parser.add_argument(
        "--query-manifest",
        type=Path,
        default=None,
        help="Optional JSONL manifest to source stable query ids from (expects --query-id-field).",
    )
    parser.add_argument(
        "--text-field",
        default="text",
        help="Text field name for optional resolution from data.hf.",
    )
    parser.add_argument(
        "--resolve-text",
        action="store_true",
        help="Attach resolved text for top-k training docs.",
    )
    return parser.parse_args(argv)
