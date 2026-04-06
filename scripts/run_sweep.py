"""Unified sweep orchestrator for operator-learning experiments.

Runs all combinations of (task, model, capacity, N, R, seeds).
"""
from __future__ import annotations

import argparse
import itertools
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from scaling_operator_learning.config_loader import load_experiment_config
from scaling_operator_learning.models import CAPACITY_GRID, available_models
from scaling_operator_learning.tasks import get_task


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


if __name__ == "__main__":
    main()
