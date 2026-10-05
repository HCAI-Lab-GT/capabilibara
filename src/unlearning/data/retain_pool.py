"""Periodic retain resampling during training.

Draws a fresh stratified retain set every K optimizer steps from the
filtered Arrow cache. Each resample uses a deterministic seed
(base_seed + optimizer_step) for reproducibility.

With NGDiff processing both retain and forget at every step, a fixed
retain set lets the model memorize the retain examples. Resampling
forces genuine capability preservation.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


@dataclass
class RetainPoolConfig:
    docs_per_bin: int = 1_000
    resample_interval: int = 200
    max_length: int = 2048


class RetainPool:
    def __init__(
        self,
        ds,
        tokenizer: PreTrainedTokenizerBase | None,
        target_topics: set[str],
        config: RetainPoolConfig,
        base_seed: int = 42,
    ) -> None:
        self.ds = ds
        self.tokenizer = tokenizer
        self.target_topics = target_topics
        self.config = config
        self.base_seed = base_seed
        self._last_resample_step = -1

    def should_resample(self, optimizer_step: int) -> bool:
        if optimizer_step == 0:
            return False
        if optimizer_step == self._last_resample_step:
            return False
        return optimizer_step % self.config.resample_interval == 0

    def resample(self, optimizer_step: int) -> list[dict]:
        """Draw a fresh stratified retain set and tokenize it."""
        from unlearning.data.dolma_pool import _tokenize
        from unlearning.data.sampling import sample_retain_stratified

        if self.tokenizer is None:
            raise ValueError("RetainPool requires a tokenizer before resampling.")

        seed = self.base_seed + optimizer_step
        rng = random.Random(seed)

        texts, _, per_topic = sample_retain_stratified(
            self.ds,
            exclude_topics=self.target_topics,
            docs_per_bin=self.config.docs_per_bin,
            rng=rng,
        )

        samples = _tokenize(texts, self.tokenizer, self.config.max_length)

        self._last_resample_step = optimizer_step
        logger.info(
            "Resampled retain at optimizer_step=%d: %d docs (seed=%d)",
            optimizer_step,
            len(samples),
            seed,
        )
        return samples
