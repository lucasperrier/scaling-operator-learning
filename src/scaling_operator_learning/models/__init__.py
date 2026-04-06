from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation: {name}")


class MLP(nn.Module):
    """Fully-connected network with configurable width, depth, and activation."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_widths: list[int],
        activation: str = "gelu",
    ):
        super().__init__()
        act = _activation(activation)
        layers: list[nn.Module] = []
        prev = in_dim
        for w in hidden_widths:
            layers.append(nn.Linear(prev, w))
            layers.append(act)
            prev = w
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


CAPACITY_GRID: dict[str, list[int]] = {
    "tiny": [32, 32],
    "small": [64, 64],
    "small-med": [96, 96],
    "medium": [128, 128],
    "med-large": [192, 192],
    "large": [256, 256],
    "xlarge": [256, 256, 256],
}


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: dict[str, Callable] = {}


def register_model(name: str):
    """Decorator to register a model builder function."""
    def decorator(func: Callable):
        _MODEL_REGISTRY[name] = func
        return func
    return decorator


def get_model(name: str) -> Callable:
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[name]


def available_models() -> list[str]:
    return list(_MODEL_REGISTRY.keys())


# Import model modules so they self-register
from . import mlp_baseline, deeponet, fno, mlp_controlled  # noqa: E402, F401
