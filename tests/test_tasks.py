"""Smoke tests for task data generation."""
from __future__ import annotations

import torch

from scaling_operator_learning.tasks import get_task, available_tasks


def test_task_registry():
    tasks = available_tasks()
    assert "burgers_operator" in tasks


def test_burgers_operator_basic():
    gen = get_task("burgers_operator")
    data = gen(n_samples=5, resolution=32, seed=42)

    assert "inputs" in data
    assert "outputs" in data
    assert "grid" in data
    assert data["resolution"] == 32
    assert data["inputs"].shape == (5, 32)
    assert data["outputs"].shape == (5, 32)
    assert data["grid"].shape == (32,)


def test_burgers_operator_resolution_varies():
    gen = get_task("burgers_operator")
    d32 = gen(n_samples=3, resolution=32, seed=42)
    d64 = gen(n_samples=3, resolution=64, seed=42)

    assert d32["inputs"].shape == (3, 32)
    assert d64["inputs"].shape == (3, 64)


def test_burgers_operator_reproducible():
    gen = get_task("burgers_operator")
    d1 = gen(n_samples=3, resolution=32, seed=99)
    d2 = gen(n_samples=3, resolution=32, seed=99)

    assert torch.allclose(d1["inputs"], d2["inputs"])
    assert torch.allclose(d1["outputs"], d2["outputs"])


def test_burgers_operator_outputs_differ():
    """Output should be different from input (the PDE evolves)."""
    gen = get_task("burgers_operator")
    data = gen(n_samples=3, resolution=64, seed=42)
    assert not torch.allclose(data["inputs"], data["outputs"], atol=1e-3)
