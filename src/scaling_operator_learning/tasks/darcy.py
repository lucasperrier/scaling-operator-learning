"""Darcy flow operator-learning task.

Generates (a, u) pairs for the 2-D Darcy flow equation (solved on 1-D slice):
    -∇·(a(x) ∇u(x)) = f(x),   x ∈ [0, 1]
    u(0) = u(1) = 0

The permeability field a(x) is a random log-normal field (exponentiated
truncated Fourier series).  The forcing is fixed: f(x) = 1.
The forward solve uses a second-order finite-difference method.
"""
from __future__ import annotations

import numpy as np
import torch

from . import register_task


def _random_permeability(
    grid: np.ndarray, rng: np.random.Generator, n_modes: int = 8, length_scale: float = 0.2
) -> np.ndarray:
    """Sample a log-normal permeability field via truncated Fourier series."""
    log_a = np.zeros_like(grid)
    for k in range(1, n_modes + 1):
        decay = np.exp(-0.5 * (k * length_scale) ** 2)
        a_k = rng.normal(0, decay)
        b_k = rng.normal(0, decay)
        log_a += a_k * np.sin(2 * np.pi * k * grid) + b_k * np.cos(2 * np.pi * k * grid)
    return np.exp(log_a)


def _fd_solve(a: np.ndarray, f: np.ndarray, dx: float) -> np.ndarray:
    """Solve -d/dx(a(x) du/dx) = f with Dirichlet BCs using finite differences.

    Uses a conservative discretization on the interior nodes:
        -1/dx^2 [ a_{i+1/2}(u_{i+1} - u_i) - a_{i-1/2}(u_i - u_{i-1}) ] = f_i
    """
    R = len(a)
    # Harmonic average for inter-cell permeability
    a_plus = 2.0 * a[:-1] * a[1:] / (a[:-1] + a[1:])  # a_{i+1/2}

    # Interior nodes (excluding boundaries which are u=0)
    n_int = R - 2
    if n_int < 1:
        return np.zeros(R)

    # Build tridiagonal system
    diag = np.zeros(n_int)
    upper = np.zeros(n_int - 1)
    lower = np.zeros(n_int - 1)
    rhs = f[1:-1] * dx ** 2

    for i in range(n_int):
        j = i + 1  # index in full grid
        diag[i] = a_plus[j - 1] + a_plus[j] if j < len(a_plus) else a_plus[j - 1]
        if j < len(a_plus):
            diag[i] = a_plus[j - 1] + a_plus[j]
        else:
            diag[i] = a_plus[j - 1]
        if i > 0:
            lower[i - 1] = -a_plus[j - 1]
        if i < n_int - 1:
            upper[i] = -a_plus[j]

    # Thomas algorithm for tridiagonal solve
    u_int = _thomas(lower, diag, upper, rhs)
    u = np.zeros(R)
    u[1:-1] = u_int
    return u


def _thomas(lower, diag, upper, rhs):
    """Thomas algorithm for tridiagonal system."""
    n = len(diag)
    c = np.zeros(n)
    d = np.zeros(n)

    c[0] = upper[0] / diag[0] if n > 1 else 0.0
    d[0] = rhs[0] / diag[0]

    for i in range(1, n):
        m = diag[i] - lower[i - 1] * c[i - 1]
        if abs(m) < 1e-15:
            m = 1e-15
        c[i] = upper[i] / m if i < n - 1 else 0.0
        d[i] = (rhs[i] - lower[i - 1] * d[i - 1]) / m

    x = np.zeros(n)
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


@register_task("darcy")
def generate_dataset(
    n_samples: int,
    resolution: int,
    seed: int,
    *,
    n_modes: int = 8,
    length_scale: float = 0.2,
) -> dict:
    """Generate Darcy flow operator-learning dataset.

    Returns:
        dict with keys:
            inputs:  Tensor (n_samples, resolution) — permeability a(x)
            outputs: Tensor (n_samples, resolution) — solution u(x)
            grid:    Tensor (resolution,) — spatial grid [0, 1]
            resolution: int
    """
    rng = np.random.default_rng(seed)
    grid = np.linspace(0, 1, resolution)
    dx = grid[1] - grid[0]
    f = np.ones(resolution)  # constant forcing

    inputs = np.zeros((n_samples, resolution), dtype=np.float64)
    outputs = np.zeros((n_samples, resolution), dtype=np.float64)

    for i in range(n_samples):
        a = _random_permeability(grid, rng, n_modes=n_modes, length_scale=length_scale)
        u = _fd_solve(a, f, dx)
        inputs[i] = a
        outputs[i] = u

    return {
        "inputs": torch.tensor(inputs, dtype=torch.float32),
        "outputs": torch.tensor(outputs, dtype=torch.float32),
        "grid": torch.tensor(grid, dtype=torch.float32),
        "resolution": resolution,
    }
