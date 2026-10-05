# pyright: reportCallIssue=false
"""Base unlearning trainer (ported/adapted from locuslab/open-unlearning)."""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import Trainer

logger = logging.getLogger(__name__)


class UnlearnTrainer(Trainer):
    """
    Base class for all unlearning trainers.

    Expects batches to be dicts with two keys:
        "forget": standard HF batch for the forget set
        "retain": standard HF batch for the retain set

    Subclasses override `training_step` to implement the unlearning algorithm.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------ #
    # Override to accept our ForgetRetainDataset's collation format        #
    # ------------------------------------------------------------------ #
    def compute_loss(
        self,
        model,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ):
        """
        Default: standard language-model loss on forget set only.
        NGDiff and other subclasses override `training_step` instead.
        """
        forget = inputs["forget"]
        output = model(**forget)
        loss = output.loss
        return (loss, output) if return_outputs else loss

    # ------------------------------------------------------------------ #
    # Prediction / evaluation loop                                         #
    # ------------------------------------------------------------------ #
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Run eval on forget set only (for PPL tracking)."""
        forget = inputs.get("forget", inputs)
        with torch.no_grad():
            output = model(**forget)
        loss = output.loss.detach()
        return (loss, None, None)
