"""Frequency-filtered diffusion task (Tier 3.4).

Identical to the standard diffusion task, but with a low-pass filter applied
to the input initial conditions.  This tests the causal mechanism behind
"resolution hurts": is it extraneous high-frequency content?

At high R, the grid can resolve frequencies up to R/2.  If the output-relevant
spectral content lives at low frequencies, the extra high-frequency modes are
noise that the model must learn to ignore.  Low-pass filtering the inputs
removes these modes, giving the model only what's relevant.

If filtering fixes the degradation at high R → strong support for
"extraneous frequencies hurt" interpretation.
"""
from __future__ import annotations

import numpy as np
import torch

from . import register_task
from .diffusion import _random_ic, _spectral_solve


def _lowpass_filter(u: np.ndarray, cutoff_modes: int) -> np.ndarray:
    """Zero out Fourier modes above cutoff_modes."""
    u_hat = np.fft.fft(u)
    R = len(u)
    freqs = np.fft.fftfreq(R, d=1.0 / R)
    mask = np.abs(freqs) <= cutoff_modes
    u_hat_filtered = u_hat * mask
    return np.fft.ifft(u_hat_filtered).real


@register_task("diffusion_filtered")
def generate_dataset(
    n_samples: int,
    resolution: int,
    seed: int,
    *,
    kappa: float = 0.01,
    T: float = 1.0,
    n_modes: int = 10,
    cutoff_modes: int | None = None,
) -> dict:
    """Generate diffusion dataset with low-pass filtered inputs.

    The cutoff_modes parameter controls the maximum retained frequency.
    If None, defaults to the number of IC modes (n_modes), which matches
    the output-relevant spectral content for diffusion (since higher
    modes decay exponentially via exp(-kappa * k^2 * T)).

    Args:
        n_samples: number of (u_0, u_T) pairs
        resolution: grid resolution R
        seed: random seed
        kappa: diffusion coefficient
        T: final time
        n_modes: number of IC Fourier modes
        cutoff_modes: low-pass filter cutoff (default: n_modes)

    Returns:
        dict with inputs, outputs, grid, resolution
    """
    if cutoff_modes is None:
        cutoff_modes = n_modes

    rng = np.random.default_rng(seed)
    grid = np.linspace(0, 2 * np.pi, resolution, endpoint=False)

    inputs = np.zeros((n_samples, resolution), dtype=np.float64)
    outputs = np.zeros((n_samples, resolution), dtype=np.float64)

    for i in range(n_samples):
        u0 = _random_ic(grid, rng, n_modes=n_modes)
        # Apply low-pass filter to remove high-frequency content
        u0_filtered = _lowpass_filter(u0, cutoff_modes)
        uT = _spectral_solve(u0_filtered, kappa=kappa, T=T)
        inputs[i] = u0_filtered
        outputs[i] = uT

    return {
        "inputs": torch.tensor(inputs, dtype=torch.float32),
        "outputs": torch.tensor(outputs, dtype=torch.float32),
        "grid": torch.tensor(grid, dtype=torch.float32),
        "resolution": resolution,
    }
