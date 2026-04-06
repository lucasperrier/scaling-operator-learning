"""Cross-resolution transfer analysis.

Evaluates models trained at resolution R_train on data at resolution R_eval,
and assembles transfer matrices showing how error degrades or improves
across resolutions.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import fit_power_law, bootstrap_power_law


def build_transfer_matrix(
    df: pd.DataFrame,
    *,
    model_name: str,
    metric_col: str = "test_rel_l2_mean",
    train_res_col: str = "resolution",
    eval_res_col: str = "eval_resolution",
) -> pd.DataFrame:
    """Build an R_train × R_eval transfer matrix for a single model.

    Args:
        df: grouped metrics DataFrame with both train and eval resolution columns.
        model_name: which model to filter.

    Returns:
        DataFrame with train resolutions as index and eval resolutions as columns.
    """
    sub = df[df["model_name"] == model_name]
    if eval_res_col not in sub.columns:
        raise ValueError(f"Column '{eval_res_col}' not found — run cross-resolution eval first.")

    pivot = sub.pivot_table(
        values=metric_col,
        index=train_res_col,
        columns=eval_res_col,
        aggfunc="mean",
    )
    return pivot.sort_index(axis=0).sort_index(axis=1)


def resolution_transfer_gain(transfer_matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute relative gain vs diagonal (same-resolution baseline).

    Returns a matrix where entry (R_train, R_eval) is:
        E(R_train, R_train) / E(R_train, R_eval) - 1
    Positive = evaluating at R_eval is better than R_train (transfer helps).
    """
    diag = np.diag(transfer_matrix.values)
    # Broadcast: diag[i] / matrix[i, j] - 1
    gain = (diag[:, None] / transfer_matrix.values) - 1.0
    return pd.DataFrame(gain, index=transfer_matrix.index, columns=transfer_matrix.columns)
