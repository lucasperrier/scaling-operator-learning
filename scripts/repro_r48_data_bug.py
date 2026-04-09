"""Standalone reproducer for the R=48 Burgers data-generator bug.

The Burgers spectral solver in `src/scaling_operator_learning/tasks/burgers.py`
produces NaN/Inf in roughly 1–4% of generated samples *only* when the spatial
resolution R = 48. Every other resolution in the main grid (16, 32, 64, 96,
128, 192, 256, 384, 512) is clean. This script reproduces the failure without
needing torch, so it can run anywhere.

Discovered while building VALIDATION_REPORT.md (Phase 2). The 582 "diverged
training runs" attributed to optimizer instability in the previous paper draft
are in fact NaN-poisoned input data caught at the first forward pass — not an
optimization failure at all.

Usage:
    python3 scripts/repro_r48_data_bug.py
"""

from __future__ import annotations

import warnings

import numpy as np

# Suppress the overflow warnings — they ARE the bug we're showing.
warnings.filterwarnings("ignore", category=RuntimeWarning)


def random_ic(grid: np.ndarray, rng: np.random.Generator, n_modes: int = 10) -> np.ndarray:
    """Inlined copy of `_random_ic` from burgers.py."""
    u = np.zeros_like(grid)
    for k in range(1, n_modes + 1):
        a_k = rng.normal(0, 1.0 / k ** 2)
        b_k = rng.normal(0, 1.0 / k ** 2)
        u += a_k * np.sin(k * grid) + b_k * np.cos(k * grid)
    return u


def spectral_solve(u0: np.ndarray, nu: float = 0.01, T: float = 1.0, dt: float = 1e-3) -> np.ndarray:
    """Inlined copy of `_spectral_solve` from burgers.py."""
    R = len(u0)
    k = np.fft.fftfreq(R, d=1.0 / R)
    L = -nu * k ** 2
    dealias = np.ones(R)
    kmax = R // 3
    dealias[np.abs(k) > kmax] = 0.0
    n_steps = max(int(np.ceil(T / dt)), 1)
    h = T / n_steps
    E_half = np.exp(L * h * 0.5)
    E_full = np.exp(L * h)

    def NL(uh):
        uhd = dealias * uh
        u = np.fft.ifft(uhd).real
        ux = np.fft.ifft(1j * k * uhd).real
        return -dealias * np.fft.fft(u * ux)

    uh = np.fft.fft(u0)
    for _ in range(n_steps):
        k1 = h * NL(uh)
        k2 = h * NL(E_half * uh + 0.5 * k1)
        k3 = h * NL(E_half * uh + 0.5 * k2)
        k4 = h * NL(E_full * uh + E_half * k3)
        uh = E_full * uh + (E_full * k1 + 2 * E_half * (k2 + k3) + k4) / 6.0
    return np.fft.ifft(uh).real


def sweep(resolutions: list[int], seeds: list[int], n_samples: int = 200) -> None:
    print(f"Sampling n_samples={n_samples} per (R, seed):")
    print()
    print(f"{'R':>4}  {'seed':>5}  {'nan_samples':>12}  {'max|u_T|':>12}")
    print("-" * 40)
    for R in resolutions:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            grid = np.linspace(0, 2 * np.pi, R, endpoint=False)
            n_nan = 0
            max_abs = 0.0
            for _ in range(n_samples):
                u0 = random_ic(grid, rng)
                uT = spectral_solve(u0)
                if np.isnan(uT).any() or np.isinf(uT).any():
                    n_nan += 1
                else:
                    max_abs = max(max_abs, float(np.abs(uT).max()))
            marker = "  <-- BUG" if n_nan > 0 else ""
            print(
                f"{R:>4}  {seed:>5}  {n_nan:>5} / {n_samples:<4}  "
                f"{max_abs:>12.3e}{marker}"
            )


def main() -> None:
    resolutions = [16, 32, 48, 64, 96, 128, 192, 256]
    seeds = [11, 22]
    sweep(resolutions, seeds)
    print()
    print(
        "Expected: only R=48 produces NaN samples. All other R values are "
        "clean. The training loop catches the NaN at the first forward pass "
        "and reports failure_reason='nan_or_inf' with best_epoch=-1, which "
        "is recorded in runs/runs_aggregate.csv as `diverged=True` for the "
        "582 affected runs at R=48."
    )


if __name__ == "__main__":
    main()
