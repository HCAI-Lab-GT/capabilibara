"""Attribution runner for precomputed query gradients.

This module loads query gradient tensors, joins them with a Bergson
training index, and writes ranked influence outputs to Parquet or JSONL.
"""

from __future__ import annotations

import logging
from pathlib import Path

from data_attribution.attribution.gradients import load_query_gradients
from data_attribution.attribution.index import (
    _import_attributor,
    _is_bergson_index,
    _load_index_gradients,
    _load_index_ids,
)
from data_attribution.attribution.metadata import load_metadata
from data_attribution.attribution.records import (
    _materialize_ids,
    _metadata_payload,
    _records_from_scores,
    persist_outputs,
)
from data_attribution.attribution.scoring import _score_query, _score_with_gradient_sets
from data_attribution.attribution.types import AttributionRunConfig


def run_attribution(config: AttributionRunConfig) -> list[dict[str, object]]:
    """Run Bergson attribution for each query gradient."""

    logger = logging.getLogger(__name__)
    metadata_lookup = load_metadata(config.metadata_path, config.doc_id_field)

    if _is_bergson_index(config.query_gradients):
        logger.info("Loading training gradients from %s", config.index_path)
        _, train_grads = _load_index_gradients(
            config.index_path, device=config.device, unit_norm=config.unit_norm
        )
        train_ids = _materialize_ids(
            _load_index_ids(config.index_path, config.doc_id_field, logger),
            next(iter(train_grads.values())).shape[0],
        )

        logger.info("Loading query gradients from %s", config.query_gradients)
        _, query_grads = _load_index_gradients(
            config.query_gradients, device=config.device, unit_norm=config.unit_norm
        )
        query_ids = _materialize_ids(
            _load_index_ids(config.query_gradients, config.query_id_field, logger),
            next(iter(query_grads.values())).shape[0],
        )

        all_records: list[dict[str, object]] = []
        total_queries = len(query_ids)
        batch_size = config.batch_size

        for start_idx in range(0, total_queries, batch_size):
            end_idx = min(start_idx + batch_size, total_queries)
            logger.info(
                "Processing query batch %d-%d of %d", start_idx, end_idx, total_queries
            )

            query_grads_batch = {
                k: v[start_idx:end_idx] for k, v in query_grads.items()
            }
            query_ids_batch = query_ids[start_idx:end_idx]

            batch_scores = _score_with_gradient_sets(
                train_grads, query_grads_batch, k=config.top_k, logger=logger
            )
            all_records.extend(
                _records_from_scores(
                    batch_scores,
                    config=config,
                    metadata_lookup=metadata_lookup,
                    training_ids=train_ids,
                    query_ids=query_ids_batch,
                )
            )

        return all_records

    logger.info("Loading query gradients from %s", config.query_gradients)
    gradients = load_query_gradients(config.query_gradients)
    logger.info("Loaded %d query gradients", len(gradients))

    logger.info("Loading attributor from %s", config.index_path)
    Attributor = _import_attributor()
    attributor = Attributor(
        config.index_path, device=config.device, unit_norm=config.unit_norm
    )

    records: list[dict[str, object]] = []
    timestamp = config.started_at.isoformat()
    for query in gradients:
        indices, scores = _score_query(attributor, query.gradient, config.top_k)
        for rank, (doc_id, score) in enumerate(zip(indices, scores), start=1):
            payload: dict[str, object] = {
                "run_id": config.run_id,
                "timestamp": timestamp,
                "query_id": query.query_id,
                "doc_id": doc_id,
                "influence_score": score,
                "rank": rank,
            }
            metadata = _metadata_payload(
                doc_id, metadata_lookup, config.metadata_join_keys
            )
            if metadata is not None:
                payload["metadata"] = metadata
            records.append(payload)
    return records


def run_and_store(config: AttributionRunConfig) -> Path:
    """Execute attribution and persist results."""

    records = run_attribution(config)
    return persist_outputs(records, config)


__all__ = [
    "AttributionRunConfig",
    "load_metadata",
    "load_query_gradients",
    "persist_outputs",
    "run_and_store",
    "run_attribution",
]
