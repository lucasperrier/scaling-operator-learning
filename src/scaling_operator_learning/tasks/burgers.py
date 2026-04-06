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
    """Sample a random initial condition as a truncated Fourier series.

    Coefficients decay as 1/k^2 to ensure smooth ICs that don't blow up
    during time evolution with low viscosity.
    """
    u = np.zeros_like(grid)
    for k in range(1, n_modes + 1):
        a_k = rng.normal(0, 1.0 / k ** 2)
        b_k = rng.normal(0, 1.0 / k ** 2)
        u += a_k * np.sin(k * grid) + b_k * np.cos(k * grid)
    return u


def _spectral_solve(u0: np.ndarray, nu: float, T: float, dt: float = 2e-4) -> np.ndarray:
    """Solve Burgers' equation with pseudo-spectral integrating-factor RK4.

    Uses an exponential integrating factor for the stiff linear (diffusion) term
    so the explicit RK4 only advances the nonlinear advection.
    Applies 2/3-rule dealiasing to prevent spectral blow-up.
    Assumes periodic domain [0, 2*pi] with R grid points.
    """
    R = len(u0)
    k = np.fft.fftfreq(R, d=1.0 / R)  # wavenumbers
    L = -nu * k ** 2  # linear operator in Fourier space

    # 2/3 dealiasing mask
    dealias = np.ones(R)
    kmax = R // 3
    dealias[np.abs(k) > kmax] = 0.0

    n_steps = max(int(np.ceil(T / dt)), 1)
    h = T / n_steps

    E_half = np.exp(L * h * 0.5)
    E_full = np.exp(L * h)

    def NL(u_hat):
        """Nonlinear term: -FFT(u * u_x) with dealiasing."""
        u_hat_d = dealias * u_hat
        u = np.fft.ifft(u_hat_d).real
        ux = np.fft.ifft(1j * k * u_hat_d).real
        return -dealias * np.fft.fft(u * ux)

    u_hat = np.fft.fft(u0)
    for _ in range(n_steps):
        k1 = h * NL(u_hat)
        k2 = h * NL(E_half * u_hat + 0.5 * k1)
        k3 = h * NL(E_half * u_hat + 0.5 * k2)
        k4 = h * NL(E_full * u_hat + E_half * k3)
        u_hat = E_full * u_hat + (E_full * k1 + 2 * E_half * (k2 + k3) + k4) / 6.0

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
