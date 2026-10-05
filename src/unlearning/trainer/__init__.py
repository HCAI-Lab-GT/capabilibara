"""Trainer registry for unlearning methods."""

from .base import UnlearnTrainer
from .ngdiff import NGDiff

TRAINER_REGISTRY = {
    "ngdiff": NGDiff,
}

__all__ = ["TRAINER_REGISTRY", "NGDiff", "UnlearnTrainer"]
