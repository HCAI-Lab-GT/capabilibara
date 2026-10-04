"""Streaming top-k influence computation using bergson Attributor.

Computes gradient-based influence scores between index examples and queries,
supporting both single-query full attribution and multi-query top-k modes.
"""

from __future__ import annotations

import argparse
import logging
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_attribution.attribution.index import _import_attributor

logger = logging.getLogger(__name__)


class RunningStats:
    """Accumulates streaming statistics (count, sum, min, max) for score arrays."""

    __slots__ = ("count", "max", "min", "sum")

    def __init__(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.min = np.inf
        self.max = -np.inf

    def update(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        self.count += int(x.size)
        self.sum += float(x.sum(dtype=np.float64))
        x_min = float(x.min())
        x_max = float(x.max())
        if x_min < self.min:
            self.min = x_min
        if x_max > self.max:
            self.max = x_max

    @property
    def mean(self) -> float:
        return self.sum / self.count if self.count > 0 else np.nan

    def as_dict(self, label: str) -> dict[str, object]:
        return {
            "label": label,
            "count": int(self.count),
            "min": float(self.min) if self.count > 0 else np.nan,
            "mean": float(self.mean),
            "max": float(self.max) if self.count > 0 else np.nan,
        }


@torch.no_grad()
def compute_topk_streaming(args: Namespace) -> None:
    """Compute influence scores with streaming top-k or single-query full mode."""
    out_dir = Path(args.attribution_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Attributions will be saved to: %s", out_dir)

    Attributor = _import_attributor()
    index = Attributor(
        Path(args.index_dataset_path), device=args.device, unit_norm=True
    )
    queries = Attributor(
        Path(args.query_dataset_path), device=args.device, unit_norm=True
    )
    logger.info("Loaded index dataset from: %s", args.index_dataset_path)

    index_grads = index.grads
    query_grads = queries.grads

    common_modules = [m for m in query_grads.keys() if m in index_grads]
    if not common_modules:
        raise ValueError("No overlapping modules between index and query grads.")

    first = common_modules[0]
    N = index_grads[first].shape[0]
    B0 = query_grads[first]
    Q = 1 if B0.ndim == 1 else B0.shape[0]

    logger.info(
        "N=%d index examples | Q=%d queries | modules=%d", N, Q, len(common_modules)
    )

    stats = RunningStats()

    if Q == 1 and args.mode == "single":
        scores_sum = None
        for name in common_modules:
            A = index_grads[name]
            B = query_grads[name].view(1, -1)
            s = (A @ B.T).squeeze(1)
            scores_sum = s if scores_sum is None else (scores_sum + s)

        scores = scores_sum
        if args.mean_across_modules:
            scores = scores / float(len(common_modules))

        scores_np = scores.float().cpu().numpy()
        stats.update(scores_np)

        pd.DataFrame(
            {"index_example_idx": np.arange(N), "attribution": scores_np}
        ).to_csv(out_dir / "attributions.csv", index=False)
        logger.info("Saved: %s", out_dir / "attributions.csv")

        pd.DataFrame([stats.as_dict("all_scores")]).to_csv(
            out_dir / "score_stats.csv", index=False
        )
        logger.info("Saved: %s", out_dir / "score_stats.csv")
        logger.info(
            "Score stats: min=%.6g mean=%.6g max=%.6g", stats.min, stats.mean, stats.max
        )
        return

    if args.mode != "topk":
        raise ValueError(
            "Your query index appears to contain multiple queries (Q>1). "
            "Use --mode topk (recommended) or rebuild queries reduced to one vector."
        )

    k = args.top_k
    blockN = args.block_n
    blockQ = args.block_q

    top_scores = np.full((Q, k), -np.inf, dtype=np.float32)
    top_indices = np.full((Q, k), -1, dtype=np.int64)

    for q0 in range(0, Q, blockQ):
        q1 = min(Q, q0 + blockQ)
        bq = q1 - q0
        logger.info("Processing queries [%d:%d) ...", q0, q1)

        block_top_scores = np.full((bq, k), -np.inf, dtype=np.float32)
        block_top_indices = np.full((bq, k), -1, dtype=np.int64)

        for i0 in range(0, N, blockN):
            i1 = min(N, i0 + blockN)
            bn = i1 - i0

            s_block = None
            for name in common_modules:
                A = index_grads[name][i0:i1]
                B = query_grads[name]
                if B.ndim == 1:
                    B = B.unsqueeze(0)
                B = B[q0:q1]
                s = A @ B.T
                s_block = s if s_block is None else (s_block + s)

            if args.mean_across_modules:
                s_block = s_block / float(len(common_modules))

            s_block_np = s_block.float().cpu().numpy()
            stats.update(s_block_np)

            cand_k = min(k, bn)
            for j in range(bq):
                col = s_block_np[:, j]
                idx = np.argpartition(-col, cand_k - 1)[:cand_k]
                vals = col[idx]

                merged_vals = np.concatenate([block_top_scores[j], vals]).astype(
                    np.float32, copy=False
                )
                merged_idx = np.concatenate([block_top_indices[j], idx + i0]).astype(
                    np.int64, copy=False
                )

                keep = np.argpartition(-merged_vals, k - 1)[:k]
                keep = keep[np.argsort(-merged_vals[keep])]
                block_top_scores[j] = merged_vals[keep]
                block_top_indices[j] = merged_idx[keep]

        top_scores[q0:q1] = block_top_scores
        top_indices[q0:q1] = block_top_indices

    rows = []
    for q in range(Q):
        for rank in range(k):
            rows.append(
                {
                    "query_idx": q,
                    "rank": rank + 1,
                    "index_example_idx": int(top_indices[q, rank]),
                    "attribution": float(top_scores[q, rank]),
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "topk_attributions.csv", index=False)
    logger.info("Saved: %s", out_dir / "topk_attributions.csv")

    pd.DataFrame([stats.as_dict("all_scores")]).to_csv(
        out_dir / "score_stats.csv", index=False
    )
    logger.info("Saved: %s", out_dir / "score_stats.csv")
    logger.info(
        "Score stats: min=%.6g mean=%.6g max=%.6g", stats.min, stats.mean, stats.max
    )


def main() -> None:
    """CLI entry point for influence computation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(
        description="Compute gradient-based influence scores using bergson."
    )
    p.add_argument("--index_dataset_path", type=Path, required=True)
    p.add_argument("--query_dataset_path", type=Path, required=True)
    p.add_argument("--attribution_path", type=Path, required=True)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--mean_across_modules", action="store_true")
    p.add_argument("--mode", type=str, default="topk", choices=["topk", "single"])
    p.add_argument("--top_k", type=int, default=100)
    p.add_argument("--block_n", type=int, default=4096, help="index block size")
    p.add_argument("--block_q", type=int, default=128, help="query block size")

    args = p.parse_args()
    compute_topk_streaming(args)


if __name__ == "__main__":
    main()
