"""Smoke tests for scaling-law analysis utilities."""
from __future__ import annotations

import numpy as np

from scaling_operator_learning.analysis import (
    fit_power_law,
    fit_full_surface,
    fit_full_volume,
    bootstrap_power_law,
)


def test_fit_power_law_basic():
    # Synthetic: E = 0.01 + 2 * N^{-0.5}
    N = np.array([50, 100, 200, 500, 1000, 2000, 5000], dtype=float)
    E = 0.01 + 2.0 * N ** (-0.5)
    fit = fit_power_law(N, E)
    assert fit is not None
    assert abs(fit["alpha"] - 0.5) < 0.05
    assert fit["E_inf"] < 0.05


def test_fit_power_law_too_few():
    assert fit_power_law(np.array([1.0, 2.0]), np.array([1.0, 0.5])) is None


def test_fit_full_surface():
    rng = np.random.default_rng(42)
    N = rng.choice([50, 100, 500, 1000], size=20).astype(float)
    D = rng.choice([100, 500, 1000, 5000], size=20).astype(float)
    E = 0.01 + 1.0 * N ** (-0.4) + 0.5 * D ** (-0.3) + rng.normal(0, 0.001, 20)
    fit = fit_full_surface(N, D, E)
    assert fit is not None
    assert "alpha" in fit and "beta" in fit


def test_fit_full_volume():
    rng = np.random.default_rng(42)
    N = rng.choice([50, 100, 500, 1000], size=30).astype(float)
    D = rng.choice([100, 500, 1000], size=30).astype(float)
    R = rng.choice([32, 64, 128, 256], size=30).astype(float)
    E = 0.01 + 1.0 * N ** (-0.4) + 0.5 * D ** (-0.3) + 0.3 * R ** (-0.5) + rng.normal(0, 0.001, 30)
    fit = fit_full_volume(N, D, R, E)
    assert fit is not None
    assert "gamma" in fit


def test_bootstrap_power_law():
    N = np.array([50, 100, 200, 500, 1000, 2000], dtype=float)
    E = 0.01 + 2.0 * N ** (-0.5)
    boot = bootstrap_power_law(N, E, n_boot=50, seed=42)
    assert boot["n_boot_success"] > 0
    assert "alpha_ci_lo" in boot
    assert "alpha_ci_hi" in boot
