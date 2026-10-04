# pyright: reportPrivateImportUsage=false
"""Loss utilities for unlearning trainers."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Standard next-token prediction loss, ignoring -100 labels."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def get_batch_loss(output, labels: torch.Tensor) -> torch.Tensor:
    """Extract loss from model output or compute from logits."""
    if hasattr(output, "loss") and output.loss is not None:
        return output.loss
    return cross_entropy_loss(output.logits, labels)


def compute_perplexity(model, dataloader, device, max_batches: int = 50) -> float:
    """Compute mean perplexity over a dataloader (no grad).

    Computes cross-entropy manually to avoid the HF internal logits.float()
    upcast which doubles peak VRAM on the logits tensor.
    """
    torch.cuda.empty_cache()
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = cross_entropy_loss(out.logits, labels)
            total_loss += loss.item()
            del out, loss
            total_batches += 1
    torch.cuda.empty_cache()
    if total_batches == 0:
        return float("inf")
    return math.exp(total_loss / total_batches)


def shuffle_tokens(
    input_ids: torch.Tensor,
    pad_token_id: int = 1,
    max_chunk_len: int = 10,
) -> torch.Tensor:
    """Chunk-shuffle each sequence in the batch (for PPL stopping criterion).

    Implements the randomized-text method from Bu & Xu (NAACL 2025):
      1. Split valid tokens into contiguous chunks of random length in [1, max_chunk_len].
      2. Randomly permute the chunk order.
      3. Paste back - preserves local n-gram structure within each chunk
         while disrupting cross-chunk semantics and long-range coherence.

    This matches the paper more closely than pure token-level shuffling,
    which over-destroys local n-gram context and yields an artificially high
    PPL threshold.
    """
    shuffled = input_ids.clone()
    for i in range(shuffled.size(0)):
        seq = shuffled[i]
        mask = seq != pad_token_id
        valid_tokens = seq[mask]
        n = valid_tokens.size(0)
        if n == 0:
            continue

        # Split into random-length chunks
        chunks: list[torch.Tensor] = []
        pos = 0
        while pos < n:
            chunk_len = torch.randint(1, max_chunk_len + 1, ()).item()
            chunks.append(valid_tokens[pos : pos + chunk_len])
            pos += chunk_len

        # Shuffle chunk order and paste back
        perm = torch.randperm(len(chunks))
        shuffled_tokens = torch.cat([chunks[p] for p in perm])
        seq[mask] = shuffled_tokens

    return shuffled


def shuffle_segments(
    input_ids: torch.Tensor,
    pad_token_id: int = 1,
    n_factor: int = 10,
) -> torch.Tensor:
    """GRACE-style segment shuffle for PPL stopping criterion.

    Splits each sequence into segments of length up to
    max(1, valid_token_count // n_factor), then shuffles segment order.
    Larger segments than shuffle_tokens: preserves multi-sentence fragments
    while destroying document-level semantic structure.

    From Zhao et al. (ACL 2024 Findings, arXiv 2402.11537).
    """
    shuffled = input_ids.clone()
    for i in range(shuffled.size(0)):
        seq = shuffled[i]
        mask = seq != pad_token_id
        valid_tokens = seq[mask]
        n = valid_tokens.size(0)
        if n == 0:
            continue

        seg_max = max(1, n // n_factor)
        segments: list[torch.Tensor] = []
        pos = 0
        while pos < n:
            seg_len = torch.randint(1, seg_max + 1, ()).item()
            segments.append(valid_tokens[pos : pos + seg_len])
            pos += seg_len

        perm = torch.randperm(len(segments))
        shuffled_tokens = torch.cat([segments[p] for p in perm])
        seq[mask] = shuffled_tokens

    return shuffled


def per_document_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Length-normalized per-document loss.

    Computes mean cross-entropy per document, then averages across the batch.
    Each sequence in the batch is treated as one document (no packing).

    logits: (B, T, V)
    labels: (B, T) with -100 for padding tokens
    Returns: scalar loss tensor
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    B, T, V = shift_logits.shape

    token_losses = F.cross_entropy(
        shift_logits.view(-1, V),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(B, T)

    valid_mask = shift_labels != -100
    valid_counts = valid_mask.sum(dim=1).float().clamp(min=1)
    doc_losses = (token_losses * valid_mask.float()).sum(dim=1) / valid_counts

    return doc_losses.mean()
