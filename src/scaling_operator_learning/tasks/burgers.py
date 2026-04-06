"""Burgers operator-learning task.

Generates (u_0, u_T) pairs for the 1-D viscous Burgers' equation:
    u_t + u * u_x = nu * u_xx
    x in [0, 2*pi],  periodic BCs,  t in [0, T]

Initial conditions are random truncated Fourier series.
The forward solve uses a pseudo-spectral method (FFT) with RK4 time-stepping.
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


def _spectral_solve(u0: np.ndarray, nu: float, T: float, dt: float = 1e-3) -> np.ndarray:
    """Solve Burgers' equation with pseudo-spectral method + RK4.

    Assumes periodic domain [0, 2*pi] with R grid points.
    """
    R = len(u0)
    k = np.fft.fftfreq(R, d=1.0 / R)  # wavenumbers

    def rhs(u_hat):
        u = np.fft.ifft(u_hat).real
        ux = np.fft.ifft(1j * k * u_hat).real
        return -np.fft.fft(u * ux) + nu * (-(k ** 2)) * u_hat

    u_hat = np.fft.fft(u0)
    n_steps = int(np.ceil(T / dt))
    dt_actual = T / n_steps

    for _ in range(n_steps):
        k1 = rhs(u_hat)
        k2 = rhs(u_hat + 0.5 * dt_actual * k1)
        k3 = rhs(u_hat + 0.5 * dt_actual * k2)
        k4 = rhs(u_hat + dt_actual * k3)
        u_hat = u_hat + (dt_actual / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return np.fft.ifft(u_hat).real


@register_task("burgers_operator")
def generate_dataset(
    n_samples: int,
    resolution: int,
    seed: int,
    *,
    nu: float = 0.01,
    T: float = 1.0,
    n_modes: int = 10,
    dt: float = 1e-3,
) -> dict:
    """Generate Burgers operator-learning dataset.

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
        uT = _spectral_solve(u0, nu=nu, T=T, dt=dt)
        inputs[i] = u0
        outputs[i] = uT

    return {
        "inputs": torch.tensor(inputs, dtype=torch.float32),
        "outputs": torch.tensor(outputs, dtype=torch.float32),
        "grid": torch.tensor(grid, dtype=torch.float32),
        "resolution": resolution,
    }
