"""Scoring helpers for attribution."""

from __future__ import annotations

import logging
from collections.abc import Mapping

import torch


def _ensure_device(
    tensor: torch.Tensor, device: str | torch.device | None
) -> torch.Tensor:
    if device is None:
        return tensor
    target = torch.device(device)
    if tensor.device != target:
        return tensor.to(target)
    return tensor


def _score_query(
    attributor: object,
    gradient: torch.Tensor,
    k: int,
) -> tuple[list[object], list[float]]:
    gradient = _ensure_device(gradient, getattr(attributor, "device", None))

    if hasattr(attributor, "score_gradients"):
        result = attributor.score_gradients(gradient, k=k)  # type: ignore[attr-defined]
    elif hasattr(attributor, "trace_from_gradients"):
        tracer = attributor.trace_from_gradients(gradient, k=k)  # type: ignore[attr-defined]
        if tracer is None:
            raise RuntimeError("Attributor.trace_from_gradients returned None")
        with tracer as result:
            pass
    else:
        raise RuntimeError(
            "Attributor must expose score_gradients or trace_from_gradients"
        )

    indices = getattr(result, "indices", [])
    scores = getattr(result, "scores", [])
    return list(indices)[:k], [float(score) for score in scores][:k]


def _score_with_gradient_sets(
    training_grads: Mapping[str, torch.Tensor],
    query_grads: Mapping[str, torch.Tensor],
    *,
    k: int,
    logger: logging.Logger,
) -> torch.Tensor:
    common_keys = sorted(set(training_grads) & set(query_grads))
    if not common_keys:
        raise ValueError("No overlapping gradient keys between training and query sets")

    scores: torch.Tensor | None = None
    for key in common_keys:
        train_tensor = training_grads[key]
        query_tensor = query_grads[key]
        if train_tensor.shape[1] != query_tensor.shape[1]:
            raise ValueError(
                f"Gradient dimension mismatch for {key}: {train_tensor.shape} vs {query_tensor.shape}"
            )
        shard = train_tensor @ query_tensor.T
        scores = shard if scores is None else scores + shard

    assert scores is not None
    if scores.shape[0] < k:
        logger.debug(
            "top_k clipped from %d to %d based on training size", k, scores.shape[0]
        )
    return scores


__all__ = ["_score_query", "_score_with_gradient_sets"]
