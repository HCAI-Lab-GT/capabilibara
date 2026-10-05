#!/usr/bin/env python3
"""Clean Dolma JSONL: drop blank-text and short-token documents."""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import time
from pathlib import Path

import zstandard as zstd

logger = logging.getLogger(__name__)

LOG_INTERVAL = 50_000


def iter_jsonl_zst_streaming(path: Path):
    dctx = zstd.ZstdDecompressor()
    with path.open("rb") as fh:
        with dctx.stream_reader(fh) as reader:
            yield from io.TextIOWrapper(reader, encoding="utf-8")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        yield from fh


def iter_hf_zst_streaming(repo_id: str, filename: str):
    """Stream a JSONL file from HuggingFace using the datasets library.

    Uses ``datasets.load_dataset(streaming=True)`` which handles HF's Xet
    storage backend transparently, unlike raw HTTP requests.
    """
    from datasets import load_dataset

    logger.info("Streaming from HuggingFace: %s/%s", repo_id, filename)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    ds = load_dataset(
        repo_id,
        data_files=filename,
        streaming=True,
        split="train",
        token=token,
    )
    for record in ds:
        yield json.dumps(record, ensure_ascii=False)


def iter_input(path: Path):
    if path.suffix == ".zst":
        return iter_jsonl_zst_streaming(path)
    return iter_jsonl(path)


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def is_blank_text(value: object) -> bool:
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    return value.strip() == ""


def token_count(text: str, tokenizer: object) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def word_count(text: str) -> int:
    return len(text.split())


def build_writer(path: Path, compress: bool):
    if compress:
        cctx = zstd.ZstdCompressor(level=3, threads=-1)
        raw = path.open("wb")
        return cctx.stream_writer(raw), raw
    raw = path.open("w", encoding="utf-8")
    return raw, raw


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Clean Dolma JSONL: drop blank and short-token documents."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        type=Path,
        help="Input JSONL or JSONL.ZST file path (local)",
    )
    input_group.add_argument(
        "--input-hf",
        nargs=2,
        metavar=("REPO_ID", "FILENAME"),
        help=(
            "Stream input from HuggingFace without downloading. "
            "Example: --input-hf HCAI-Lab/archive-dolma3-pool-150b archive-dolma3-pool-150b.jsonl.zst"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output file path",
    )
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument(
        "--min-tokens",
        type=int,
        default=None,
        help="Drop documents with fewer tokens (uses HF tokenizer, slow)",
    )
    filter_group.add_argument(
        "--min-words",
        type=int,
        default=None,
        help="Drop documents with fewer whitespace-split words (fast, no tokenizer)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="allenai/OLMo-3-1025-7B",
        help="HF tokenizer model ID (only used with --min-tokens)",
    )
    parser.add_argument(
        "--compress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write output as zst-compressed JSONL (default: True)",
    )
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    tokenizer = None
    if args.min_tokens:
        logger.info("Loading tokenizer: %s", args.tokenizer)
        tokenizer = load_tokenizer(args.tokenizer)
        logger.info("Tokenizer loaded")
    elif args.min_words:
        logger.info(
            "Using word-count filter (min_words=%d) — no tokenizer needed",
            args.min_words,
        )
    else:
        logger.info("No length filter — only dropping blank/parse-error documents")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    dropped_blank = 0
    dropped_short = 0
    dropped_parse = 0
    total = 0
    start = time.monotonic()

    writer, raw_handle = build_writer(args.output, args.compress)

    try:
        if args.input_hf:
            line_iter = iter_hf_zst_streaming(args.input_hf[0], args.input_hf[1])
        else:
            line_iter = iter_input(args.input)

        for line in line_iter:
            total += 1
            line = line.strip()
            if not line:
                dropped_blank += 1
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                dropped_parse += 1
                continue

            text = obj.get("text")
            if is_blank_text(text):
                dropped_blank += 1
                continue

            if args.min_tokens and token_count(text, tokenizer) < args.min_tokens:
                dropped_short += 1
                continue

            if args.min_words and word_count(text) < args.min_words:
                dropped_short += 1
                continue

            out_line = json.dumps(obj, ensure_ascii=False) + "\n"
            if args.compress:
                writer.write(out_line.encode("utf-8"))
            else:
                writer.write(out_line)
            kept += 1

            if total % LOG_INTERVAL == 0:
                elapsed = max(time.monotonic() - start, 1e-6)
                logger.info(
                    "Progress: %s total, %s kept, %s dropped "
                    "(blank=%s, short=%s, parse=%s) %.0f docs/s",
                    total,
                    kept,
                    dropped_blank + dropped_short + dropped_parse,
                    dropped_blank,
                    dropped_short,
                    dropped_parse,
                    total / elapsed,
                )
    finally:
        if args.compress:
            writer.flush(zstd.FLUSH_FRAME)
        writer.close()
        if raw_handle is not writer:
            raw_handle.close()

    elapsed = time.monotonic() - start
    input_label = (
        f"hf://{args.input_hf[0]}/{args.input_hf[1]}"
        if args.input_hf
        else str(args.input)
    )
    logger.info("Input file  : %s", input_label)
    logger.info("Output file : %s", args.output)
    logger.info("Total rows  : %s", total)
    logger.info("Kept        : %s", kept)
    logger.info("Dropped     : %s", dropped_blank + dropped_short + dropped_parse)
    logger.info("  blank     : %s", dropped_blank)
    short_label = (
        f"< {args.min_tokens} tokens"
        if args.min_tokens
        else (f"< {args.min_words} words" if args.min_words else "n/a")
    )
    logger.info("  short     : %s (%s)", dropped_short, short_label)
    logger.info("  parse err : %s", dropped_parse)
    logger.info("Elapsed     : %.1f s (%.0f docs/s)", elapsed, total / max(elapsed, 1))


if __name__ == "__main__":
    main()
