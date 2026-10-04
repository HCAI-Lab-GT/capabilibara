from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import torch


def select_gradient_keys(
    training_grads: Mapping[str, torch.Tensor],
    query_grads: Mapping[str, torch.Tensor],
    preferred: str | None,
) -> Sequence[str]:
    if preferred is not None:
        if preferred not in training_grads:
            raise KeyError(f"Gradient key '{preferred}' missing from training grads")
        if preferred not in query_grads:
            raise KeyError(f"Gradient key '{preferred}' missing from query grads")
        return [preferred]

    overlap = sorted(set(training_grads) & set(query_grads))
    if not overlap:
        raise ValueError(
            "No overlapping gradient keys between training and query indices"
        )
    return overlap


def score_batch(
    training_grads: Mapping[str, torch.Tensor],
    query_grads: Mapping[str, torch.Tensor],
    keys: Sequence[str],
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    combined: torch.Tensor | None = None
    for key in keys:
        train_tensor = training_grads[key]
        query_tensor = query_grads[key]
        if train_tensor.shape[1] != query_tensor.shape[1]:
            raise ValueError(
                "Gradient dimension mismatch for "
                f"{key}: {train_tensor.shape} vs {query_tensor.shape}"
            )
        shard = train_tensor @ query_tensor.T
        combined = shard if combined is None else combined + shard

    if combined is None:
        raise ValueError("No gradients were provided for scoring")

    top_k = min(k, combined.shape[0])
    scores, indices = torch.topk(combined, k=top_k, dim=0)
    return scores, indices


def batched_topk(
    training_grads: Mapping[str, torch.Tensor],
    query_grads: Mapping[str, torch.Tensor],
    *,
    keys: Sequence[str],
    k: int,
    batch_size: int,
) -> Iterable[tuple[int, torch.Tensor, torch.Tensor]]:
    total_queries = next(iter(query_grads.values())).shape[0]
    for start in range(0, total_queries, batch_size):
        end = min(start + batch_size, total_queries)
        batch_slice = slice(start, end)
        batch_queries = {
            key: tensor[batch_slice] for key, tensor in query_grads.items()
        }
        scores, indices = score_batch(training_grads, batch_queries, keys, k)
        yield start, scores, indices
