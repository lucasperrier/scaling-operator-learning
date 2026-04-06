"""Controlled-parameter MLP for the resolution confound experiment (Tier 3.2).

Like the standard MLP baseline, this maps R -> hidden -> R, but the hidden
width is automatically adjusted so that the total parameter count stays
approximately constant across different resolutions.

For a 2-layer MLP with hidden widths [h, h]:
    D = R*h + h + h*h + h + h*R + R = 2*R*h + h^2 + 2*h + R

Given a target_params (set at a reference resolution, e.g., R=64 with
medium capacity), we solve for h at each resolution R.
"""
from __future__ import annotations

import math

import torch.nn as nn

from . import MLP, register_model, CAPACITY_GRID


def _solve_hidden_width(target_params: int, resolution: int, n_hidden_layers: int = 2) -> int:
    """Find the hidden width h such that a 2-layer MLP R->h->h->R has ≈ target_params.

    For n_hidden_layers=2: D = 2*R*h + h^2 + 2*h + R
    Rearranging: h^2 + (2*R + 2)*h + (R - target_params) = 0

    For n_hidden_layers=3: D = R*h + h + h*h + h + h*h + h + h*R + R = 2*R*h + 2*h^2 + 3*h + R
    Rearranging: 2*h^2 + (2*R + 3)*h + (R - target_params) = 0
    """
    if n_hidden_layers == 2:
        a = 1
        b = 2 * resolution + 2
        c = resolution - target_params
    elif n_hidden_layers == 3:
        a = 2
        b = 2 * resolution + 3
        c = resolution - target_params
    else:
        raise ValueError(f"Only n_hidden_layers=2 or 3 supported, got {n_hidden_layers}")

    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return 8  # Minimum fallback

    h = (-b + math.sqrt(discriminant)) / (2 * a)
    h = max(8, round(h))  # Floor at 8 to avoid degenerate networks
    return h


def _compute_mlp_params(resolution: int, hidden_widths: list[int]) -> int:
    """Compute exact parameter count for MLP(R -> hidden -> R)."""
    total = 0
    prev = resolution
    for w in hidden_widths:
        total += prev * w + w  # weight + bias
        prev = w
    total += prev * resolution + resolution  # output layer
    return total


# Pre-compute target parameter counts for each capacity level at reference R=64
_REFERENCE_R = 64
_CONTROLLED_TARGETS: dict[str, int] = {}
for _cap_name, _widths in CAPACITY_GRID.items():
    _CONTROLLED_TARGETS[_cap_name] = _compute_mlp_params(_REFERENCE_R, _widths)


@register_model("mlp_controlled")
def build_mlp_controlled(
    resolution: int,
    hidden_widths: list[int] | None = None,
    activation: str = "gelu",
    capacity_name: str = "medium",
    **kwargs,
) -> nn.Module:
    """Build an MLP with approximately fixed parameter count across resolutions.

    The hidden width is computed to match the parameter count of the standard
    MLP capacity at R=64 (the reference resolution). This disentangles the
    resolution axis from the capacity axis.

    Args:
        resolution: input/output dimension (grid resolution)
        hidden_widths: ignored (overridden by controlled width computation)
        activation: activation function name
        capacity_name: capacity level (determines target param count)
    """
    ref_widths = CAPACITY_GRID.get(capacity_name, CAPACITY_GRID["medium"])
    n_layers = len(ref_widths)
    target_params = _CONTROLLED_TARGETS.get(capacity_name, _CONTROLLED_TARGETS["medium"])

    h = _solve_hidden_width(target_params, resolution, n_hidden_layers=n_layers)
    controlled_widths = [h] * n_layers

    return MLP(
        in_dim=resolution,
        out_dim=resolution,
        hidden_widths=controlled_widths,
        activation=activation,
    )
