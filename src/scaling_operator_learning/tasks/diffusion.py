"""Diffusion operator-learning task.

Generates (u_0, u_T) pairs for the 1-D heat/diffusion equation:
    u_t = kappa * u_xx
    x ∈ [0, 2π],  periodic BCs,  t ∈ [0, T]

Initial conditions are random truncated Fourier series.
The forward solve is exact in Fourier space: each mode decays as
    û_k(t) = û_k(0) * exp(-kappa * k^2 * t)
"""
from __future__ import annotations

import numpy as np
import torch

from . import register_task


def _random_ic(grid: np.ndarray, rng: np.random.Generator, n_modes: int = 10) -> np.ndarray:
    """Sample a random initial condition as a truncated Fourier series."""
    u = np.zeros_like(grid)
    for k in range(1, n_modes + 1):
        a_k = rng.normal(0, 1.0 / k)
        b_k = rng.normal(0, 1.0 / k)
        u += a_k * np.sin(k * grid) + b_k * np.cos(k * grid)
    return u


def _spectral_solve(u0: np.ndarray, kappa: float, T: float) -> np.ndarray:
    """Solve the heat equation exactly in Fourier space.

    Each Fourier mode decays exponentially:  û_k(T) = û_k(0) * exp(-kappa * k^2 * T)
    """
    R = len(u0)
    k = np.fft.fftfreq(R, d=1.0 / R)  # wavenumbers
    u_hat = np.fft.fft(u0)
    decay = np.exp(-kappa * k ** 2 * T)
    u_hat_T = u_hat * decay
    return np.fft.ifft(u_hat_T).real


@register_task("diffusion")
def generate_dataset(
    n_samples: int,
    resolution: int,
    seed: int,
    *,
    kappa: float = 0.01,
    T: float = 1.0,
    n_modes: int = 10,
) -> dict:
    """Generate diffusion operator-learning dataset.

    Returns:
        dict with keys:
            inputs:  Tensor (n_samples, resolution) — u_0 values on grid
            outputs: Tensor (n_samples, resolution) — u_T values on grid
            grid:    Tensor (resolution,) — spatial grid points
            resolution: int
    """
    rng = np.random.default_rng(seed)
    grid = np.linspace(0, 2 * np.pi, resolution, endpoint=False)

    inputs = np.zeros((n_samples, resolution), dtype=np.float64)
    outputs = np.zeros((n_samples, resolution), dtype=np.float64)

    for i in range(n_samples):
        u0 = _random_ic(grid, rng, n_modes=n_modes)
        uT = _spectral_solve(u0, kappa=kappa, T=T)
        inputs[i] = u0
        outputs[i] = uT

    return {
        "inputs": torch.tensor(inputs, dtype=torch.float32),
        "outputs": torch.tensor(outputs, dtype=torch.float32),
        "grid": torch.tensor(grid, dtype=torch.float32),
        "resolution": resolution,
    }
