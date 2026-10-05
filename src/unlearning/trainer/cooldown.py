"""Retain-only cooldown manager.

During cooldown phases, the forget gradient pass is skipped and only
retain gradients are applied. This gives the model periods of pure
retain optimization without competing forget pressure.

Disabled by default (--cooldown_enabled false).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CooldownConfig:
    enabled: bool = False
    every_k: int = 200
    steps: int = 50


class CooldownManager:
    def __init__(self, config: CooldownConfig) -> None:
        self.config = config
        self._in_cooldown = False
        self._cooldown_remaining = 0
        self._last_trigger_step = -1

    def should_cooldown(self, optimizer_step: int) -> bool:
        if not self.config.enabled:
            return False
        if self._in_cooldown:
            return False
        if optimizer_step == 0:
            return False
        return optimizer_step % self.config.every_k == 0

    def enter_cooldown(self, optimizer_step: int) -> None:
        self._in_cooldown = True
        self._cooldown_remaining = self.config.steps
        self._last_trigger_step = optimizer_step
        logger.info(
            "Entering cooldown at optimizer_step=%d for %d steps",
            optimizer_step,
            self.config.steps,
        )

    def step(self) -> None:
        if not self._in_cooldown:
            return
        self._cooldown_remaining -= 1
        if self._cooldown_remaining <= 0:
            self._in_cooldown = False
            logger.info("Exiting cooldown after %d steps", self.config.steps)

    def in_cooldown(self) -> bool:
        return self._in_cooldown

    @property
    def remaining_steps(self) -> int:
        return self._cooldown_remaining
