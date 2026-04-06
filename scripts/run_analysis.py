"""Run scaling analysis on aggregated results and save fit summaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scaling_operator_learning.analysis import run_scaling_analysis


def _clean_for_json(obj):
    """Recursively remove non-serializable items (e.g. numpy arrays)."""
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_for_json(v) for v in obj]
    if isinstance(obj, float):
        return obj
    if isinstance(obj, (int, str, bool, type(None))):
        return obj
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)


def main():
    parser = argparse.ArgumentParser(description="Run scaling-law analysis")
    parser.add_argument("--grouped-csv", type=str, required=True,
                        help="Path to grouped_metrics.csv")
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.grouped_csv)
    print(f"Loaded {len(df)} groups from {args.grouped_csv}")

    fits = run_scaling_analysis(df, metric_col="test_rel_l2_mean", n_boot=args.n_boot)

    out_path = out_dir / "scaling_fits.json"
    with open(out_path, "w") as f:
        json.dump(_clean_for_json(fits), f, indent=2)
    print(f"Saved fits → {out_path}")

    # Summary
    print("\nData-scaling exponents (α) by model:")
    for rec in fits["data_fits"]:
        alpha = rec.get("alpha")
        boot = rec.get("bootstrap", {})
        if alpha is not None:
            lo = boot.get("alpha_ci_lo", float("nan"))
            hi = boot.get("alpha_ci_hi", float("nan"))
            res_str = f" R={rec.get('resolution', '?')}" if "resolution" in rec else ""
            print(
                f"  {rec['model_name']:16s} cap={rec.get('capacity_name', '?'):10s}"
                f"{res_str}  α={alpha:.3f}  [{lo:.3f}, {hi:.3f}]"
            )

    if fits.get("resolution_fits"):
        print("\nResolution-scaling exponents (γ) by model:")
        for rec in fits["resolution_fits"]:
            alpha = rec.get("alpha")
            boot = rec.get("bootstrap", {})
            if alpha is not None:
                lo = boot.get("alpha_ci_lo", float("nan"))
                hi = boot.get("alpha_ci_hi", float("nan"))
                print(
                    f"  {rec['model_name']:16s} cap={rec['capacity_name']:10s} "
                    f"N={rec['dataset_size']:<6}  γ={alpha:.3f}  [{lo:.3f}, {hi:.3f}]"
                )

    if fits.get("full_3d_fits"):
        print("\nFull 3D fits E(N,D,R):")
        for rec in fits["full_3d_fits"]:
            alpha = rec.get("alpha")
            beta = rec.get("beta")
            gamma = rec.get("gamma")
            if alpha is not None:
                print(
                    f"  {rec['model_name']:16s}  α={alpha:.3f}  β={beta:.3f}  γ={gamma:.3f}"
                )


if __name__ == "__main__":
    main()
