"""Unified sweep orchestrator for operator-learning experiments.

Runs all combinations of (task, model, capacity, N, R, seeds).
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from scaling_operator_learning.config_loader import load_experiment_config
from scaling_operator_learning.models import CAPACITY_GRID, available_models, get_model, parameter_count
from scaling_operator_learning.models.fno import FNO_CAPACITY_GRID
from scaling_operator_learning.tasks import get_task
from scaling_operator_learning.training.train import _forward
from scaling_operator_learning.utils import save_json


def _run_single(job: dict) -> dict:
    # Import here to avoid pickling issues with spawn
    from scaling_operator_learning.training.train import train_one_run
    return train_one_run(**job)


def main():
    parser = argparse.ArgumentParser(description="Operator-learning experiment sweep")
    parser.add_argument("--config", default="configs/burgers_operator.yaml")
    parser.add_argument("--task", default=None, help="Task name (overrides config)")
    parser.add_argument("--models", default="mlp_baseline,deeponet,fno")
    parser.add_argument("--capacities", default=None, help="CSV capacity names")
    parser.add_argument("--dataset-sizes", default=None, help="CSV of N values")
    parser.add_argument("--resolutions", default=None, help="CSV of R values")
    parser.add_argument("--data-seeds", default=None)
    parser.add_argument("--train-seeds", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--pilot", action="store_true", help="Small pilot grid")
    parser.add_argument("--device", default=None, help="Force device: cpu or cuda")
    parser.add_argument("-j", "--jobs", type=int, default=1)
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)

    task_name = args.task or cfg.task.name
    models = args.models.split(",")
    capacities = args.capacities.split(",") if args.capacities else list(CAPACITY_GRID.keys())
    dataset_sizes = (
        [int(x) for x in args.dataset_sizes.split(",")]
        if args.dataset_sizes else cfg.data.n_train_sizes
    )
    resolutions = (
        [int(x) for x in args.resolutions.split(",")]
        if args.resolutions else cfg.resolution.train_resolutions
    )
    eval_resolutions = cfg.resolution.eval_resolutions
    data_seeds = (
        [int(x) for x in args.data_seeds.split(",")]
        if args.data_seeds else cfg.data.data_seeds
    )
    train_seeds = (
        [int(x) for x in args.train_seeds.split(",")]
        if args.train_seeds else cfg.train.train_seeds
    )

    if args.pilot:
        models = models[:2]
        capacities = capacities[:3]
        dataset_sizes = dataset_sizes[:3]
        resolutions = resolutions[:2]
        data_seeds = data_seeds[:1]
        train_seeds = train_seeds[:1]

    out_root = Path(cfg.out_dir)
    gen_dataset = get_task(task_name)

    # Pre-generate data per (resolution, data_seed) to avoid redundant solves
    print(f"Task: {task_name}")
    print(f"Models: {models}")
    print(f"Capacities: {capacities}")
    print(f"N: {dataset_sizes}")
    print(f"R: {resolutions}")
    print(f"Seeds: data={data_seeds}, train={train_seeds}")
    print()

    print("Pre-generating datasets...")
    data_cache: dict[tuple[int, int], dict] = {}
    max_n = max(dataset_sizes)
    n_test = cfg.data.n_test

    for R, ds_seed in itertools.product(resolutions, data_seeds):
        train_data = gen_dataset(n_samples=max_n, resolution=R, seed=ds_seed)
        test_data = gen_dataset(n_samples=n_test, resolution=R, seed=ds_seed + 10000)
        data_cache[(R, ds_seed)] = {"train": train_data, "test": test_data}
    print(f"  Cached {len(data_cache)} (R, seed) combinations\n")

    # Build and run jobs
    t0 = time.time()
    n_total = 0
    n_success = 0
    n_skipped = 0

    for ds_seed in data_seeds:
        seed_jobs: list[dict] = []
        for model, cap, N, R, ts in itertools.product(
            models, capacities, dataset_sizes, resolutions, train_seeds
        ):
            run_dir = (
                out_root / f"task={task_name}" / f"model={model}" / f"capacity={cap}"
                / f"N={N}" / f"R={R}" / f"data_seed={ds_seed}" / f"seed={ts}"
            )

            if not args.overwrite and (run_dir / "metrics.json").exists():
                n_skipped += 1
                continue

            cached = data_cache[(R, ds_seed)]
            job = {
                "model_tag": model,
                "capacity_name": cap,
                "resolution": R,
                "dataset_size": N,
                "train_seed": ts,
                "train_inputs": cached["train"]["inputs"],
                "train_outputs": cached["train"]["outputs"],
                "test_inputs": cached["test"]["inputs"],
                "test_outputs": cached["test"]["outputs"],
                "grid": cached["train"]["grid"],
                "run_dir": str(run_dir),
                "lr": cfg.train.lr,
                "weight_decay": cfg.train.weight_decay,
                "max_epochs": cfg.train.max_epochs,
                "patience": cfg.train.early_stopping_patience,
                "batch_size": cfg.train.batch_size,
                "activation": cfg.model.activation,
            }
            if args.device:
                job["device"] = args.device
            seed_jobs.append(job)

        if not seed_jobs:
            continue

        print(f"[seed={ds_seed}] Launching {len(seed_jobs)} runs...")

        if args.jobs <= 1:
            for i, job in enumerate(seed_jobs):
                print(
                    f"  [{n_total + i + 1}] {job['model_tag']} "
                    f"cap={job['capacity_name']} N={job['dataset_size']} R={job['resolution']}"
                )
                r = _run_single(job)
                if r.get("status") == "success":
                    n_success += 1
        else:
            ctx = __import__("multiprocessing").get_context("spawn")
            with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx) as pool:
                results = list(pool.map(_run_single, seed_jobs))
            n_success += sum(1 for r in results if r.get("status") == "success")

        n_total += len(seed_jobs)

    elapsed = time.time() - t0
    print(f"\nDone: {n_success}/{n_total} successful, {n_skipped} skipped ({elapsed:.0f}s)")

    # === Cross-resolution evaluation ===
    # Extra eval resolutions that aren't in the training set
    extra_eval_res = [r for r in eval_resolutions if r not in resolutions]
    if not extra_eval_res:
        print("\nNo extra eval resolutions to evaluate.")
        return

    print(f"\n=== Cross-resolution evaluation at R_eval={extra_eval_res} ===")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    n_eval = 0

    for ds_seed in data_seeds:
        # Pre-generate eval data at extra resolutions
        eval_cache: dict[int, dict] = {}
        for R_eval in extra_eval_res:
            eval_cache[R_eval] = gen_dataset(
                n_samples=cfg.data.n_test, resolution=R_eval, seed=ds_seed + 10000
            )

        for model, cap, N, R_train, ts in itertools.product(
            models, capacities, dataset_sizes, resolutions, train_seeds
        ):
            train_run_dir = (
                out_root / f"task={task_name}" / f"model={model}" / f"capacity={cap}"
                / f"N={N}" / f"R={R_train}" / f"data_seed={ds_seed}" / f"seed={ts}"
            )
            best_pt = train_run_dir / "best.pt"
            if not best_pt.exists():
                continue

            # Build model to load weights
            build_fn = get_model(model)
            if model in ("fno", "mlp_controlled"):
                mdl = build_fn(resolution=R_train, capacity_name=cap)
            else:
                hidden = CAPACITY_GRID[cap]
                mdl = build_fn(resolution=R_train, hidden_widths=hidden, activation=cfg.model.activation)
            mdl.load_state_dict(torch.load(best_pt, weights_only=True, map_location=device))
            mdl = mdl.to(device)
            mdl.eval()

            for R_eval in extra_eval_res:
                eval_dir = (
                    out_root / f"task={task_name}" / f"model={model}" / f"capacity={cap}"
                    / f"N={N}" / f"R={R_train}" / f"data_seed={ds_seed}" / f"seed={ts}"
                    / f"eval_resolution={R_eval}"
                )
                if not args.overwrite and (eval_dir / "metrics.json").exists():
                    continue

                # MLP variants can't evaluate at different resolution
                if model in ("mlp_baseline", "mlp_controlled") and R_eval != R_train:
                    continue

                edata = eval_cache[R_eval]
                x_eval = edata["inputs"].to(device)
                y_eval = edata["outputs"].to(device)
                grid_eval = edata["grid"].to(device)

                with torch.no_grad():
                    if model == "deeponet" and R_eval != R_train:
                        # Branch net expects input at training resolution;
                        # interpolate eval inputs from R_eval → R_train grid
                        x_branch = torch.nn.functional.interpolate(
                            x_eval.unsqueeze(1), size=R_train,
                            mode="linear", align_corners=False,
                        ).squeeze(1)
                        x_query = grid_eval.unsqueeze(-1)
                        y_pred = mdl(x_branch, x_query)
                    else:
                        y_pred = _forward(mdl, x_eval, grid_eval, model)
                    test_rel_l2 = (torch.norm(y_pred - y_eval) / torch.norm(y_eval)).item()
                    test_mse = ((y_pred - y_eval) ** 2).mean().item()

                eval_metrics = {
                    "model_name": model,
                    "capacity_name": cap,
                    "parameter_count": parameter_count(mdl),
                    "resolution": R_train,
                    "eval_resolution": R_eval,
                    "dataset_size": N,
                    "train_seed": ts,
                    "status": "success",
                    "test_rel_l2": test_rel_l2,
                    "test_mse": test_mse,
                    "eligible_for_fit": True,
                    "run_dir": str(eval_dir),
                    "device": device,
                }
                save_json(eval_dir / "metrics.json", eval_metrics)
                n_eval += 1

    print(f"Cross-resolution eval: {n_eval} evaluations saved.")


if __name__ == "__main__":
    main()
