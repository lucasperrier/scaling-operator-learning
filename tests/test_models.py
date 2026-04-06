"""Smoke tests for model building and forward passes."""
from __future__ import annotations

import torch

from scaling_operator_learning.models import (
    MLP,
    get_model,
    available_models,
    parameter_count,
    CAPACITY_GRID,
)


def test_model_registry():
    models = available_models()
    assert "mlp_baseline" in models
    assert "deeponet" in models
    assert "fno" in models


def test_mlp_baseline_forward():
    build = get_model("mlp_baseline")
    model = build(resolution=64, hidden_widths=[32, 32], activation="gelu")
    x = torch.randn(4, 64)
    y = model(x)
    assert y.shape == (4, 64)
    assert parameter_count(model) > 0


def test_deeponet_forward():
    build = get_model("deeponet")
    model = build(resolution=64, hidden_widths=[32, 32], activation="gelu")
    u_sensors = torch.randn(4, 64)  # batch of input functions
    x_query = torch.linspace(0, 1, 64).unsqueeze(-1)  # (64, 1)
    y = model(u_sensors, x_query)
    assert y.shape == (4, 64)


def test_fno_forward():
    build = get_model("fno")
    model = build(resolution=64, capacity_name="tiny")
    u = torch.randn(4, 64)
    grid = torch.linspace(0, 1, 64)
    y = model(u, grid)
    assert y.shape == (4, 64)


def test_fno_resolution_invariance():
    """FNO should accept different resolutions without retraining."""
    build = get_model("fno")
    model = build(resolution=64, capacity_name="tiny")
    model.eval()

    # Forward at resolution 64
    u64 = torch.randn(2, 64)
    g64 = torch.linspace(0, 1, 64)
    y64 = model(u64, g64)
    assert y64.shape == (2, 64)

    # Forward at resolution 128 (different from build-time)
    u128 = torch.randn(2, 128)
    g128 = torch.linspace(0, 1, 128)
    y128 = model(u128, g128)
    assert y128.shape == (2, 128)


def test_capacity_grid_param_counts():
    """Larger capacity should have more parameters."""
    build = get_model("mlp_baseline")
    m_small = build(resolution=64, hidden_widths=CAPACITY_GRID["small"])
    m_large = build(resolution=64, hidden_widths=CAPACITY_GRID["large"])
    assert parameter_count(m_small) < parameter_count(m_large)
