"""Unified training loop for operator-learning experiments.

Model-agnostic: works with mlp_baseline, deeponet, and fno through a common
interface.  Each run:
  1. Builds the model from the registry
  2. Trains with early stopping on a validation split
  3. Evaluates on held-out test data
  4. Saves metrics.json to the run directory
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..models import get_model, parameter_count, CAPACITY_GRID
from ..models.fno import FNO_CAPACITY_GRID
from ..utils import save_json


def _forward(model: nn.Module, inputs: torch.Tensor, grid: torch.Tensor, model_tag: str) -> torch.Tensor:
    """Unified forward call that dispatches to the right signature."""
    if model_tag == "deeponet":
        # DeepONet: branch(u_sensors), trunk(x_query)
        x_query = grid.unsqueeze(-1)  # (R,) -> (R, 1)
        return model(inputs, x_query)  # (batch, R)
    elif model_tag == "fno":
        return model(inputs, grid)     # (batch, R)
    else:
        # MLP baseline: flat in -> flat out
        return model(inputs)           # (batch, R)


def train_one_run(
    *,
    model_tag: str,
    capacity_name: str,
    resolution: int,
    dataset_size: int,
    train_seed: int,
    train_inputs: torch.Tensor,
    train_outputs: torch.Tensor,
    test_inputs: torch.Tensor,
    test_outputs: torch.Tensor,
    grid: torch.Tensor,
    run_dir: str | Path,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    max_epochs: int = 5000,
    patience: int = 200,
    batch_size: int = 64,
    activation: str = "gelu",
    device: str | None = None,
    val_frac: float = 0.1,
    optimizer_type: str = "adam",
    scheduler_type: str | None = None,
) -> dict[str, Any]:
    """Train a single operator-learning run and save results.

    Args:
        optimizer_type: 'adam' (default) or 'sgd'
        scheduler_type: None (default), 'cosine', or 'step'

    Returns metrics dict (also saved to run_dir/metrics.json).
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Subset training data to requested dataset_size
    n = min(dataset_size, train_inputs.size(0))
    all_inputs = train_inputs[:n].to(device)
    all_outputs = train_outputs[:n].to(device)
    grid_dev = grid.to(device)

    # Validation split
    n_val = max(1, int(n * val_frac))
    n_train = n - n_val
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(train_seed))
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    x_train = all_inputs[train_idx]
    y_train = all_outputs[train_idx]
    x_val = all_inputs[val_idx]
    y_val = all_outputs[val_idx]

    x_test = test_inputs.to(device)
    y_test = test_outputs.to(device)

    # Build model
    build_fn = get_model(model_tag)
    if model_tag in ("fno", "mlp_controlled"):
        model = build_fn(resolution=resolution, capacity_name=capacity_name)
    else:
        hidden = CAPACITY_GRID[capacity_name]
        model = build_fn(resolution=resolution, hidden_widths=hidden, activation=activation)
    model = model.to(device)

    n_params = parameter_count(model)

    # Build optimizer
    if optimizer_type == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
        )
    else:  # adam (default)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Build scheduler
    scheduler = None
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max_epochs // 3, gamma=0.1)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    start_time = time.time()

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()

        # Mini-batch or full batch
        if n_train <= batch_size:
            y_pred = _forward(model, x_train, grid_dev, model_tag)
            loss = ((y_pred - y_train) ** 2).mean()
        else:
            idx = torch.randperm(n_train, device=device)[:batch_size]
            y_pred = _forward(model, x_train[idx], grid_dev, model_tag)
            loss = ((y_pred - y_train[idx]) ** 2).mean()

        if torch.isnan(loss) or torch.isinf(loss):
            return _failure_metrics(
                model_tag, capacity_name, resolution, n_params, dataset_size,
                train_seed, run_dir, "nan_or_inf", time.time() - start_time, device,
            )

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Validation check
        if epoch % 50 == 0 or epoch == max_epochs - 1:
            model.eval()
            with torch.no_grad():
                y_val_pred = _forward(model, x_val, grid_dev, model_tag)
                val_mse = ((y_val_pred - y_val) ** 2).mean().item()
                val_rel_l2 = (
                    torch.norm(y_val_pred - y_val) / torch.norm(y_val)
                ).item()

            if val_rel_l2 < best_val_loss:
                best_val_loss = val_rel_l2
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save(model.state_dict(), run_dir / "best.pt")
            else:
                epochs_no_improve += 50

            if epochs_no_improve >= patience:
                break

    runtime = time.time() - start_time

    # Load best model for final eval
    if (run_dir / "best.pt").exists():
        model.load_state_dict(torch.load(run_dir / "best.pt", weights_only=True))

    model.eval()
    with torch.no_grad():
        y_test_pred = _forward(model, x_test, grid_dev, model_tag)
        test_mse = ((y_test_pred - y_test) ** 2).mean().item()
        test_rel_l2 = (
            torch.norm(y_test_pred - y_test) / torch.norm(y_test)
        ).item()

    metrics = {
        "model_name": model_tag,
        "capacity_name": capacity_name,
        "parameter_count": n_params,
        "resolution": resolution,
        "dataset_size": dataset_size,
        "train_seed": train_seed,
        "status": "success",
        "failure_reason": "",
        "best_epoch": best_epoch,
        "test_rel_l2": test_rel_l2,
        "test_mse": test_mse,
        "val_rel_l2": best_val_loss,
        "runtime_seconds": runtime,
        "diverged": False,
        "nan_detected": False,
        "eligible_for_fit": True,
        "run_dir": str(run_dir),
        "device": device,
        "optimizer": optimizer_type,
        "scheduler": scheduler_type or "none",
    }

    save_json(run_dir / "metrics.json", metrics)
    return metrics


def _failure_metrics(
    model_tag, capacity_name, resolution, n_params, dataset_size,
    train_seed, run_dir, reason, runtime, device,
) -> dict[str, Any]:
    metrics = {
        "model_name": model_tag,
        "capacity_name": capacity_name,
        "parameter_count": n_params,
        "resolution": resolution,
        "dataset_size": dataset_size,
        "train_seed": train_seed,
        "status": "failed",
        "failure_reason": reason,
        "best_epoch": -1,
        "test_rel_l2": float("nan"),
        "test_mse": float("nan"),
        "val_rel_l2": float("nan"),
        "runtime_seconds": runtime,
        "diverged": reason == "nan_or_inf",
        "nan_detected": reason == "nan_or_inf",
        "eligible_for_fit": False,
        "run_dir": str(run_dir),
        "device": device,
    }
    save_json(Path(run_dir) / "metrics.json", metrics)
    return metrics
