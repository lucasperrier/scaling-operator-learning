"""DeepONet: Deep Operator Network.

Branch net encodes the input function (discretized at sensor locations).
Trunk net encodes the query location(s).
Output is the dot product of branch and trunk outputs.

Reference: Lu et al., "Learning nonlinear operators via DeepONet" (2021).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import MLP, register_model


class DeepONet(nn.Module):
    """Unstacked DeepONet with MLP branch and trunk nets."""

    def __init__(
        self,
        branch_in: int,
        trunk_in: int,
        hidden_widths: list[int],
        p: int = 64,
        activation: str = "gelu",
    ):
        super().__init__()
        self.branch = MLP(branch_in, p, hidden_widths, activation=activation)
        self.trunk = MLP(trunk_in, p, hidden_widths, activation=activation)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u_sensors: torch.Tensor, x_query: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u_sensors: (batch, n_sensors) — input function values at sensor locations
            x_query:   (n_query, trunk_in) — query coordinates

        Returns:
            (batch, n_query) — predicted output function values
        """
        b = self.branch(u_sensors)          # (batch, p)
        t = self.trunk(x_query)             # (n_query, p)
        # Dot product: (batch, p) @ (p, n_query) -> (batch, n_query)
        out = torch.einsum("bp,qp->bq", b, t) + self.bias
        return out


@register_model("deeponet")
def build_deeponet(
    resolution: int,
    hidden_widths: list[int],
    activation: str = "gelu",
    p: int | None = None,
) -> DeepONet:
    """Build a DeepONet for 1-D operator learning.

    Args:
        resolution: number of sensor points (branch input dim)
        hidden_widths: shared hidden layer widths for branch and trunk
        p: latent dimension (defaults to last hidden width)
    """
    if p is None:
        p = hidden_widths[-1] if hidden_widths else 64
    return DeepONet(
        branch_in=resolution,
        trunk_in=1,  # 1-D spatial coordinate
        hidden_widths=hidden_widths,
        p=p,
        activation=activation,
    )
