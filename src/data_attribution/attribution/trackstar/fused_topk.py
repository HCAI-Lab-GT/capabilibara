"""Memory-bounded, deterministic top-k candidate extraction for score shards."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np


class CandidateRow(TypedDict):
    query_id: str
    doc_id: str
    score: float
    rank: int


@dataclass
class TopKAccumulator:
    """Checkpointable per-query winners across streamed score batches."""

    query_ids: tuple[str, ...]
    top_k: int
    candidates: list[list[tuple[float, str]]]

    @classmethod
    def create(cls, query_ids: Sequence[str], *, top_k: int) -> TopKAccumulator:
        values = tuple(str(query_id) for query_id in query_ids)
        if not values or len(set(values)) != len(values):
            raise ValueError("query_ids must be nonempty and unique")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        return cls(
            query_ids=values,
            top_k=top_k,
            candidates=[[] for _ in values],
        )

    def update(self, scores: np.ndarray, doc_ids: Sequence[str]) -> None:
        matrix = np.asarray(scores)
        if (
            matrix.ndim != 2
            or matrix.shape[0] == 0
            or matrix.shape[1] != len(self.query_ids)
        ):
            raise ValueError("scores must match the accumulator query dimension")
        if len(doc_ids) != matrix.shape[0]:
            raise ValueError("doc_ids length must match scores document dimension")
        doc_values = [str(doc_id) for doc_id in doc_ids]
        if len(set(doc_values)) != len(doc_values):
            raise ValueError("doc_ids must be unique within a score batch")
        if not np.isfinite(matrix).all():
            raise ValueError("scores must be finite")

        local_k = min(self.top_k, len(doc_values))
        doc_array = np.asarray(doc_values, dtype=str)
        for query_index in range(matrix.shape[1]):
            column = matrix[:, query_index]
            threshold = np.partition(column, -local_k)[-local_k]
            above = np.flatnonzero(column > threshold)
            tied = np.flatnonzero(column == threshold)
            tie_slots = local_k - len(above)
            tied = tied[np.argsort(doc_array[tied], kind="stable")[:tie_slots]]
            local_indices = np.concatenate((above, tied))
            incoming = [
                (float(column[index]), doc_values[int(index)])
                for index in local_indices
            ]
            self.candidates[query_index] = sorted(
                (*self.candidates[query_index], *incoming),
                key=lambda item: (-item[0], item[1]),
            )[: self.top_k]

    def write(self, output_file: Path) -> int:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_file.with_suffix(output_file.suffix + ".tmp")
        written = 0
        with temporary.open("w", encoding="utf-8") as handle:
            for query_id, candidates in zip(
                self.query_ids, self.candidates, strict=True
            ):
                for rank, (score, doc_id) in enumerate(candidates, 1):
                    handle.write(
                        json.dumps(
                            {
                                "query_id": query_id,
                                "doc_id": doc_id,
                                "score": score,
                                "rank": rank,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    written += 1
        temporary.replace(output_file)
        return written


def iter_topk_candidates(
    scores: np.ndarray,
    doc_ids: Sequence[str],
    query_ids: Sequence[str],
    *,
    top_k: int,
) -> Iterator[CandidateRow]:
    """Yield per-query winners in source query order.

    ``scores`` is consumed one column at a time.  The only temporary arrays
    are the current column and its local candidate indices, so callers can
    discard the shard matrix as soon as iteration completes.
    """
    matrix = np.asarray(scores)
    if matrix.ndim != 2:
        raise ValueError("scores must be a two-dimensional [docs, queries] matrix")
    if len(doc_ids) != matrix.shape[0]:
        raise ValueError("doc_ids length must match scores document dimension")
    if len(query_ids) != matrix.shape[1]:
        raise ValueError("query_ids length must match scores query dimension")
    if top_k <= 0 or top_k > matrix.shape[0]:
        raise ValueError("top_k must be positive and no larger than document count")
    if len(set(map(str, doc_ids))) != len(doc_ids):
        raise ValueError("doc_ids must be unique within a shard")
    if len(set(map(str, query_ids))) != len(query_ids):
        raise ValueError("query_ids must be unique")

    doc_id_values = [str(doc_id) for doc_id in doc_ids]
    doc_array = np.asarray(doc_id_values, dtype=str)
    query_id_values = [str(query_id) for query_id in query_ids]
    for query_index, query_id in enumerate(query_id_values):
        column = matrix[:, query_index]
        threshold = np.partition(column, -top_k)[-top_k]
        above = np.flatnonzero(column > threshold)
        tied = np.flatnonzero(column == threshold)
        tie_slots = top_k - len(above)
        tied = tied[np.argsort(doc_array[tied], kind="stable")[:tie_slots]]
        local_indices = np.concatenate((above, tied))
        ordered = sorted(
            (int(index) for index in local_indices),
            key=lambda index: (-float(column[index]), doc_id_values[index]),
        )
        for rank, index in enumerate(ordered, 1):
            yield {
                "query_id": query_id,
                "doc_id": doc_id_values[index],
                "score": float(column[index]),
                "rank": rank,
            }


def write_topk_candidates(
    scores: np.ndarray,
    doc_ids: Sequence[str],
    query_ids: Sequence[str],
    output_file: Path,
    *,
    top_k: int,
) -> int:
    """Write contiguous JSONL candidate groups and return rows written."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_suffix(output_file.suffix + ".tmp")
    written = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in iter_topk_candidates(scores, doc_ids, query_ids, top_k=top_k):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    temporary.replace(output_file)
    return written
