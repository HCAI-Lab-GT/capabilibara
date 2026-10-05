from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Mapping

import torch


def import_attributor():
    spec = importlib.util.find_spec("bergson.query.attributor")
    if spec is None:
        raise RuntimeError("Install the bergson package to use Attributor.")
    module = importlib.import_module("bergson.query.attributor")
    return module.Attributor


def ensure_device(
    tensor: torch.Tensor, device: str | torch.device | None
) -> torch.Tensor:
    if device is None:
        return tensor
    target = torch.device(device)
    return tensor.to(target) if tensor.device != target else tensor


def load_index(
    index_path, device: str | None
) -> tuple[object, Mapping[str, torch.Tensor]]:
    Attributor = import_attributor()
    attributor = Attributor(index_path=index_path, device=device, unit_norm=True)
    grads = {
        name: ensure_device(tensor, device) for name, tensor in attributor.grads.items()
    }
    if not grads:
        raise ValueError(f"No gradients found under {index_path}")
    return attributor, grads
