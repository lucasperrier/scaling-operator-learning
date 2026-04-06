"""Tier 3.5 — DeepONet interpolation ablation.

Re-evaluates saved DeepONet models at cross-resolution settings using
native-resolution branch evaluation (no interpolation), vs. the
standard interpolation approach used in the main experiments.

Standard approach: when evaluating a model trained at R_train on data
at R_eval ≠ R_train, the branch input (R_eval sensors) is interpolated
to R_train sensors. The trunk net evaluates at R_eval query points.

Native approach: rebuild the DeepONet with branch_in=R_eval, load only
the trunk weights and matching internal layers, and evaluate directly
at R_eval resolution. This tests whether the interpolation step in
cross-resolution transfer introduces artifacts.

NOTE: True native-resolution eval for DeepONet requires retraining with
the target resolution. This script instead compares two evaluation
strategies for models already trained:
  (a) Interpolation (the current default)
  (b) Zero-padded / truncated branch input (approximate native eval)

Usage:
    python scripts/run_deeponet_ablation.py [--runs-root runs] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from scaling_operator_learning.models import CAPACITY_GRID, parameter_count
from scaling_operator_learning.models.deeponet import build_deeponet
from scaling_operator_learning.tasks import get_task
from scaling_operator_learning.utils import save_json


def _evaluate_interpolated(model, x_eval, grid_eval, R_train, device):
    """Standard cross-resolution: interpolate eval inputs to R_train grid."""
    x_branch = F.interpolate(
        x_eval.unsqueeze(1), size=R_train,
        mode="linear", align_corners=False,
    ).squeeze(1)
    x_query = grid_eval.unsqueeze(-1)  # (R_eval, 1)
    y_pred = model(x_branch, x_query)
    return y_pred


def _evaluate_truncated(model, x_eval, grid_eval, R_train, device):
    """Ablation: truncate or zero-pad branch input to R_train without interpolation.

    If R_eval > R_train: take the first R_train points (subsampled at lower resolution).
    If R_eval < R_train: zero-pad to R_train points.

    This tests whether interpolation artifacts matter by using a cruder but
    interpolation-free approach.
    """
    R_eval = x_eval.shape[1]
    if R_eval >= R_train:
        # Subsample: take evenly spaced R_train points from R_eval
        indices = torch.linspace(0, R_eval - 1, R_train, device=device).long()
        x_branch = x_eval[:, indices]
    else:
        # Zero-pad
        x_branch = F.pad(x_eval, (0, R_train - R_eval), mode="constant", value=0.0)

    x_query = grid_eval.unsqueeze(-1)
    y_pred = model(x_branch, x_query)
    return y_pred


def main():
    parser = argparse.ArgumentParser(description="DeepONet interpolation ablation (Tier 3.5)")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--device", default=None)
    parser.add_argument("--tasks", default="burgers_operator,darcy,diffusion")
    parser.add_argument("--capacities", default="medium,large,xlarge")
    parser.add_argument("--data-seed", type=int, default=11)
    parser.add_argument("--train-seed", type=int, default=101)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    runs_root = Path(args.runs_root)
    tasks = args.tasks.split(",")
    capacities = args.capacities.split(",")

    train_resolutions = [32, 64, 128, 256]
    eval_resolutions = [32, 64, 128, 256, 512]
    dataset_sizes = [500, 1000, 2000, 5000]

    results = []

    for task_name in tasks:
        gen_data = get_task(task_name)
        n_test = 1000

        for cap in capacities:
            hidden = CAPACITY_GRID[cap]

            for N in dataset_sizes:
                for R_train in train_resolutions:
                    run_dir = (
                        runs_root / f"task={task_name}" / "model=deeponet"
                        / f"capacity={cap}" / f"N={N}" / f"R={R_train}"
                        / f"data_seed={args.data_seed}" / f"seed={args.train_seed}"
                    )
                    best_pt = run_dir / "best.pt"
                    if not best_pt.exists():
                        continue

                    # Build and load model
                    model = build_deeponet(
                        resolution=R_train, hidden_widths=hidden, activation="gelu"
                    )
                    model.load_state_dict(
                        torch.load(best_pt, weights_only=True, map_location=device)
                    )
                    model = model.to(device)
                    model.eval()

                    for R_eval in eval_resolutions:
                        if R_eval == R_train:
                            continue  # Same-resolution eval is identical for both methods

                        eval_data = gen_data(
                            n_samples=n_test, resolution=R_eval,
                            seed=args.data_seed + 10000,
                        )
                        x_eval = eval_data["inputs"].to(device)
                        y_eval = eval_data["outputs"].to(device)
                        grid_eval = eval_data["grid"].to(device)

                        with torch.no_grad():
                            # Method (a): interpolation (standard)
                            y_interp = _evaluate_interpolated(
                                model, x_eval, grid_eval, R_train, device
                            )
                            rel_l2_interp = (
                                torch.norm(y_interp - y_eval)
                                / torch.norm(y_eval)
                            ).item()

                            # Method (b): truncation/pad (no interpolation)
                            y_trunc = _evaluate_truncated(
                                model, x_eval, grid_eval, R_train, device
                            )
                            rel_l2_trunc = (
                                torch.norm(y_trunc - y_eval)
                                / torch.norm(y_eval)
                            ).item()

                        rec = {
                            "task": task_name,
                            "capacity": cap,
                            "dataset_size": N,
                            "R_train": R_train,
                            "R_eval": R_eval,
                            "rel_l2_interpolated": rel_l2_interp,
                            "rel_l2_truncated": rel_l2_trunc,
                            "interp_better": rel_l2_interp < rel_l2_trunc,
                            "relative_difference": (
                                (rel_l2_trunc - rel_l2_interp)
                                / max(rel_l2_interp, 1e-12)
                            ),
                        }
                        results.append(rec)
                        print(
                            f"  {task_name:20s} cap={cap:8s} N={N:5d} "
                            f"R_train={R_train:3d} R_eval={R_eval:3d}: "
                            f"interp={rel_l2_interp:.4f}  trunc={rel_l2_trunc:.4f}  "
                            f"{'interp✓' if rec['interp_better'] else 'trunc✓'}"
                        )

    # Summary
    if results:
        n_interp_wins = sum(1 for r in results if r["interp_better"])
        print(f"\n{'='*80}")
        print(f"Interpolation wins: {n_interp_wins}/{len(results)}")
        print(f"Truncation wins:    {len(results) - n_interp_wins}/{len(results)}")
        mean_diff = sum(r["relative_difference"] for r in results) / len(results)
        print(f"Mean relative difference (trunc - interp) / interp: {mean_diff:.2%}")

    # Save
    out = Path("results")
    out.mkdir(exist_ok=True)
    with open(out / "deeponet_interpolation_ablation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → results/deeponet_interpolation_ablation.json")


if __name__ == "__main__":
    main()
