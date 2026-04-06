"""Discretization-tied MLP baseline for operator learning.

Takes a flattened input function (evaluated on a fixed grid of resolution R)
and outputs a flattened output function on the same grid.  The parameter count
scales with R, making this a resolution-dependent baseline.
"""
from __future__ import annotations

import torch.nn as nn

from . import MLP, register_model


@register_model("mlp_baseline")
def build_mlp_baseline(
    resolution: int,
    hidden_widths: list[int],
    activation: str = "gelu",
) -> nn.Module:
    """Build a discretization-tied MLP: R -> hidden -> R."""
    return MLP(
        in_dim=resolution,
        out_dim=resolution,
        hidden_widths=hidden_widths,
        activation=activation,
    )
