"""Tier 3.6 — Early-stopping / optimiser sensitivity ablation.

Reruns a subset of experiments (Burgers, all models, medium capacity) with:
  (a) 2× patience (400 instead of 200)
  (b) SGD + cosine schedule (instead of Adam)

Then compares law-selection winners against the baseline Adam runs.

Usage:
    python scripts/run_optimizer_ablation.py --config configs/burgers_operator.yaml \
      --device cuda -j 1
"""
from __future__ import annotations

import argparse
import itertools
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from scaling_operator_learning.config_loader import load_experiment_config
from scaling_operator_learning.models import CAPACITY_GRID
from scaling_operator_learning.tasks import get_task
from scaling_operator_learning.utils import save_json


def _run_single(job: dict) -> dict:
    from scaling_operator_learning.training.train import train_one_run
    return train_one_run(**job)


def main():
    parser = argparse.ArgumentParser(description="Optimizer sensitivity ablation (Tier 3.6)")
    parser.add_argument("--config", default="configs/burgers_operator.yaml")
    parser.add_argument("--task", default=None)
    parser.add_argument("--models", default="mlp_baseline,deeponet,fno")
    parser.add_argument("--capacities", default="medium")
    parser.add_argument("--device", default=None)
    parser.add_argument("-j", "--jobs", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    task_name = args.task or cfg.task.name
    models = args.models.split(",")
    capacities = args.capacities.split(",")
    dataset_sizes = cfg.data.n_train_sizes
    resolutions = cfg.resolution.train_resolutions
    data_seeds = cfg.data.data_seeds
    train_seeds = cfg.train.train_seeds

    gen_dataset = get_task(task_name)

    # Define ablation variants
    ablation_configs = [
        {
            "name": "adam_2x_patience",
            "optimizer_type": "adam",
            "scheduler_type": None,
            "patience": cfg.train.early_stopping_patience * 2,  # 400
            "max_epochs": cfg.train.max_epochs,
        },
        {
            "name": "sgd_cosine",
            "optimizer_type": "sgd",
            "scheduler_type": "cosine",
            "patience": cfg.train.early_stopping_patience,
            "max_epochs": cfg.train.max_epochs,
        },
    ]

    for ablation in ablation_configs:
        abl_name = ablation["name"]
        out_root = Path(cfg.out_dir + f"_ablation_{abl_name}")
        print(f"\n{'='*60}")
        print(f"Ablation: {abl_name}")
        print(f"  optimizer: {ablation['optimizer_type']}")
        print(f"  scheduler: {ablation['scheduler_type']}")
        print(f"  patience:  {ablation['patience']}")
        print(f"  out_dir:   {out_root}")
        print(f"{'='*60}\n")

        # Pre-generate data
        data_cache: dict[tuple[int, int], dict] = {}
        max_n = max(dataset_sizes)

        for R, ds_seed in itertools.product(resolutions, data_seeds):
            train_data = gen_dataset(n_samples=max_n, resolution=R, seed=ds_seed)
            test_data = gen_dataset(
                n_samples=cfg.data.n_test, resolution=R, seed=ds_seed + 10000
            )
            data_cache[(R, ds_seed)] = {"train": train_data, "test": test_data}

        t0 = time.time()
        n_total = 0
        n_success = 0

        for ds_seed in data_seeds:
            jobs: list[dict] = []
            for model, cap, N, R, ts in itertools.product(
                models, capacities, dataset_sizes, resolutions, train_seeds
            ):
                run_dir = (
                    out_root / f"task={task_name}" / f"model={model}"
                    / f"capacity={cap}" / f"N={N}" / f"R={R}"
                    / f"data_seed={ds_seed}" / f"seed={ts}"
                )
                if not args.overwrite and (run_dir / "metrics.json").exists():
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
                    "max_epochs": ablation["max_epochs"],
                    "patience": ablation["patience"],
                    "batch_size": cfg.train.batch_size,
                    "activation": cfg.model.activation,
                    "optimizer_type": ablation["optimizer_type"],
                    "scheduler_type": ablation["scheduler_type"],
                }
                if args.device:
                    job["device"] = args.device
                jobs.append(job)

            if not jobs:
                continue

            print(f"[{abl_name}] [seed={ds_seed}] Launching {len(jobs)} runs...")

            if args.jobs <= 1:
                for job in jobs:
                    r = _run_single(job)
                    if r.get("status") == "success":
                        n_success += 1
            else:
                ctx = __import__("multiprocessing").get_context("spawn")
                with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx) as pool:
                    results = list(pool.map(_run_single, jobs))
                n_success += sum(1 for r in results if r.get("status") == "success")

            n_total += len(jobs)

        elapsed = time.time() - t0
        print(f"\n[{abl_name}] Done: {n_success}/{n_total} ({elapsed:.0f}s)")
        print(f"  Results in: {out_root}")


if __name__ == "__main__":
    main()
