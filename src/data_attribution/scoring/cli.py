from __future__ import annotations

import logging
from collections.abc import Sequence

import torch

from data_attribution.scoring.args import parse_args
from data_attribution.scoring.core import batched_topk, select_gradient_keys
from data_attribution.scoring.index import load_index
from data_attribution.scoring.io import (
    build_record,
    load_id_field,
    load_text_lookup,
    materialize_ids,
    resolve_query_ids,
    write_json,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logger = logging.getLogger("batched_scoring")

    logger.info("Loading training index from %s", args.training_index)
    train_attr, training_grads = load_index(args.training_index, args.device)
    train_ids = materialize_ids(
        load_id_field(args.training_index, args.doc_id_field),
        next(iter(training_grads.values())).shape[0],
    )

    logger.info("Loading query index from %s", args.query_index)
    query_attr, query_grads = load_index(args.query_index, args.device)
    total_queries = next(iter(query_grads.values())).shape[0]
    query_ids = resolve_query_ids(
        args.query_index,
        args.query_id_field,
        args.query_manifest,
        total_queries,
    )

    gradient_keys = select_gradient_keys(training_grads, query_grads, args.gradient_key)
    logger.info("Scoring with gradient keys: %s", ", ".join(gradient_keys))

    logger.info(
        "Batch size %d implies ~%.2f MB score buffer per key for %d training docs",
        args.query_batch_size,
        (next(iter(training_grads.values())).shape[0] * args.query_batch_size * 4)
        / (1024 * 1024),
        len(train_ids),
    )

    text_lookup = (
        load_text_lookup(args.training_index, args.doc_id_field, args.text_field)
        if args.resolve_text
        else None
    )
    if args.resolve_text and text_lookup is None:
        logger.warning(
            "Text resolution requested but data.hf or fields are missing; proceeding without text"
        )

    results: list[dict[str, object]] = []
    with torch.no_grad():
        for start, scores, indices in batched_topk(
            training_grads,
            query_grads,
            keys=gradient_keys,
            k=args.top_k,
            batch_size=args.query_batch_size,
        ):
            for column, query_idx in enumerate(range(start, start + scores.shape[1])):
                query_id = query_ids[query_idx]
                column_scores = scores[:, column].tolist()
                column_indices = indices[:, column].tolist()
                doc_ids = [train_ids[i] for i in column_indices]
                results.append(
                    build_record(
                        query_id,
                        doc_ids,
                        [float(score) for score in column_scores],
                        gradient_keys,
                        text_lookup,
                    )
                )

    write_json(results, args.output)
    logger.info("Wrote %d query records to %s", len(results), args.output)
    logger.info("Training device: %s", getattr(train_attr, "device", None))
    logger.info("Query device: %s", getattr(query_attr, "device", None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
