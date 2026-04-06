"""Collect metrics.json files into aggregate and grouped CSV tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _infer_from_path(metrics_path: Path) -> dict[str, Any]:
    """Extract metadata from directory structure: task=X/model=Y/capacity=Z/N=W/R=V/..."""
    run_dir = metrics_path.parent
    inferred: dict[str, Any] = {
        "run_dir": str(run_dir.resolve()),
        "metrics_path": str(metrics_path.resolve()),
    }
    for part in run_dir.parts:
        if "=" in part:
            key, val = part.split("=", 1)
            if key in ("model", "model_name"):
                inferred.setdefault("model_name", val)
            elif key == "capacity":
                inferred.setdefault("capacity_name", val)
            elif key in ("N", "D", "dataset_size"):
                inferred.setdefault("dataset_size", int(val))
            elif key in ("R", "resolution"):
                inferred.setdefault("resolution", int(val))
            elif key == "seed":
                inferred.setdefault("train_seed", int(val))
            elif key == "data_seed":
                inferred.setdefault("data_seed", int(val))
            elif key == "train_seed":
                inferred.setdefault("train_seed", int(val))
            elif key == "task":
                inferred.setdefault("task", val)
            elif key == "eval_resolution":
                inferred.setdefault("eval_resolution", int(val))
            else:
                inferred.setdefault(key, val)
    return inferred


def collect_records(runs_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for metrics_path in sorted(runs_root.rglob("metrics.json")):
        payload = _load_json(metrics_path)
        inferred = _infer_from_path(metrics_path)
        row = {**inferred, **payload}
        row["capacity_name"] = str(row.get("capacity_name", "custom"))
        row["status"] = str(row.get("status", "success"))
        row["eligible_for_fit"] = bool(row.get("eligible_for_fit", row["status"] == "success"))
        records.append(row)
    return records


def group_records(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-run results into grouped statistics."""
    group_keys = ["model_name", "capacity_name", "parameter_count", "dataset_size"]
    for opt in ["resolution", "task", "eval_resolution"]:
        if opt in df.columns:
            group_keys.append(opt)
    group_keys = [k for k in group_keys if k in df.columns]

    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(group_keys, dropna=False):
        key_map = dict(zip(group_keys, keys if isinstance(keys, tuple) else (keys,)))
        eligible = group[group["eligible_for_fit"]] if "eligible_for_fit" in group.columns else group

        row: dict[str, Any] = {
            **key_map,
            "n_runs": int(len(eligible)),
            "n_attempted": int(len(group)),
        }
        if "diverged" in group.columns:
            row["divergence_rate"] = float(group["diverged"].astype(float).mean())

        for metric in ["test_rel_l2", "test_mse", "val_rel_l2", "runtime_seconds"]:
            if metric not in eligible.columns:
                continue
            series = pd.to_numeric(eligible[metric], errors="coerce")
            row[f"{metric}_mean"] = float(series.mean()) if not series.empty else float("nan")
            row[f"{metric}_std"] = float(series.std(ddof=1)) if len(series) > 1 else float("nan")
            row[f"{metric}_stderr"] = (
                float(series.std(ddof=1) / (len(series) ** 0.5)) if len(series) > 1 else float("nan")
            )

        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_keys).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate metrics.json files into CSVs")
    parser.add_argument("--runs-root", type=str, default="runs")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    if not runs_root.exists():
        raise FileNotFoundError(f"Runs root does not exist: {runs_root}")

    out_dir = Path(args.out_dir) if args.out_dir else runs_root
    out_dir.mkdir(parents=True, exist_ok=True)

    records = collect_records(runs_root)
    if not records:
        raise FileNotFoundError(f"No metrics.json files found under {runs_root}")

    aggregate_df = pd.DataFrame(records)
    grouped_df = group_records(aggregate_df)

    aggregate_path = out_dir / "runs_aggregate.csv"
    grouped_path = out_dir / "grouped_metrics.csv"
    aggregate_df.to_csv(aggregate_path, index=False)
    grouped_df.to_csv(grouped_path, index=False)

    print(f"Wrote {len(aggregate_df)} runs → {aggregate_path}")
    print(f"Wrote {len(grouped_df)} groups → {grouped_path}")


if __name__ == "__main__":
    main()
