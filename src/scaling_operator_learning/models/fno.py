"""Fourier Neural Operator (FNO) for 1-D operator learning.

Operates in the frequency domain via spectral convolutions.
Resolution-invariant: the same trained model can be evaluated on different grids.

Reference: Li et al., "Fourier Neural Operator for Parametric PDEs" (2021).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_model


class SpectralConv1d(nn.Module):
    """1-D spectral convolution layer."""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, R)
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(
            x.size(0), self.out_channels, x_ft.size(-1),
            dtype=torch.cfloat, device=x.device,
        )
        modes = min(self.modes, x_ft.size(-1))
        out_ft[:, :, :modes] = torch.einsum(
            "bim,iom->bom", x_ft[:, :, :modes], self.weights[:, :, :modes]
        )
        return torch.fft.irfft(out_ft, n=x.size(-1))


class FNO1d(nn.Module):
    """1-D Fourier Neural Operator."""

    def __init__(
        self,
        modes: int = 16,
        width: int = 64,
        n_layers: int = 4,
    ):
        super().__init__()
        self.modes = modes
        self.width = width

        self.lift = nn.Linear(2, width)  # (u, x) -> width

        self.spectral_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.spectral_layers.append(SpectralConv1d(width, width, modes))
            self.linear_layers.append(nn.Conv1d(width, width, 1))

        self.project = nn.Sequential(
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, u: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u:    (batch, R) — input function values
            grid: (R,) or (batch, R) — spatial coordinates

        Returns:
            (batch, R) — predicted output function values
        """
        if grid.dim() == 1:
            grid = grid.unsqueeze(0).expand(u.size(0), -1)
        # Stack input: (batch, R, 2)
        x = torch.stack([u, grid], dim=-1)
        # Lift: (batch, R, width)
        x = self.lift(x)
        # Transpose for conv: (batch, width, R)
        x = x.permute(0, 2, 1)

        for spec, lin in zip(self.spectral_layers, self.linear_layers):
            x1 = spec(x)
            x2 = lin(x)
            x = F.gelu(x1 + x2)

        # Project back: (batch, R, width) -> (batch, R, 1) -> (batch, R)
        x = x.permute(0, 2, 1)
        x = self.project(x).squeeze(-1)
        return x


# ---------------------------------------------------------------------------
# FNO capacity presets (modes, width, n_layers)
# ---------------------------------------------------------------------------
FNO_CAPACITY_GRID: dict[str, dict[str, int]] = {
    "tiny":      {"modes": 8,  "width": 16, "n_layers": 2},
    "small":     {"modes": 12, "width": 32, "n_layers": 3},
    "small-med": {"modes": 16, "width": 48, "n_layers": 3},
    "medium":    {"modes": 16, "width": 64, "n_layers": 4},
    "med-large": {"modes": 20, "width": 96, "n_layers": 4},
    "large":     {"modes": 24, "width": 128, "n_layers": 4},
    "xlarge":    {"modes": 32, "width": 128, "n_layers": 5},
}


@register_model("fno")
def build_fno(
    resolution: int,
    hidden_widths: list[int] | None = None,
    activation: str = "gelu",
    capacity_name: str = "medium",
    **kwargs,
) -> FNO1d:
    """Build an FNO1d from capacity preset.

    The hidden_widths arg is accepted for API compatibility but ignored;
    FNO capacity is controlled through its own preset grid.
    """
    preset = FNO_CAPACITY_GRID.get(capacity_name, FNO_CAPACITY_GRID["medium"])
    return FNO1d(
        modes=preset["modes"],
        width=preset["width"],
        n_layers=preset["n_layers"],
    )
