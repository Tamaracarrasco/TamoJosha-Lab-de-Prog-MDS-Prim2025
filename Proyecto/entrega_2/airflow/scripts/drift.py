# airflow/scripts/drift.py

from __future__ import annotations
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


def _compute_psi(base: np.ndarray, new: np.ndarray, buckets: int = 10) -> float:
    """
    Cálculo simple y correcto del Population Stability Index (PSI).
    """
    base = pd.Series(base).replace([np.inf, -np.inf], np.nan).dropna()
    new = pd.Series(new).replace([np.inf, -np.inf], np.nan).dropna()

    if len(base) == 0 or len(new) == 0:
        return 0.0

    quantiles = np.linspace(0, 1, buckets + 1)
    cuts = np.unique(base.quantile(quantiles).values)

    if len(cuts) <= 2:
        return 0.0

    base_counts, _ = np.histogram(base, bins=cuts)
    new_counts, _ = np.histogram(new, bins=cuts)

    base_perc = base_counts / base_counts.sum()
    new_perc = new_counts / new_counts.sum()

    eps = 1e-6
    base_perc = np.clip(base_perc, eps, 1)
    new_perc = np.clip(new_perc, eps, 1)

    psi = np.sum((base_perc - new_perc) * np.log(base_perc / new_perc))
    return float(psi)


def detect_drift_last_week(
    df_final: pd.DataFrame,
    num_cols: Optional[List[str]] = None,
    psi_threshold: float = 0.2,
    buckets: int = 10,
) -> Tuple[bool, Dict]:

    """
    Compara la última semana contra todo el histórico usando PSI.
    """

    if "semana_num" not in df_final.columns:
        raise ValueError("df_final debe contener 'semana_num'.")

    max_week = df_final["semana_num"].max()

    base_df = df_final[df_final["semana_num"] < max_week]
    new_df = df_final[df_final["semana_num"] == max_week]

    if base_df.empty or new_df.empty:
        return False, {
            "reason": "No hay semanas suficientes para comparar.",
            "semana_max": int(max_week),
        }

    # Detectamos columnas numéricas si no se pasan explícitas
    if num_cols is None:
        num_df = df_final.select_dtypes(include=["int64", "float64"])
        excluir = {"semana_num", "y"}
        num_cols = [c for c in num_df.columns if c not in excluir]

    psi_per_feature: Dict[str, float] = {}

    for col in num_cols:
        psi_per_feature[col] = _compute_psi(
            base_df[col].values,
            new_df[col].values,
            buckets=buckets
        )

    drift_detected = any(v >= psi_threshold for v in psi_per_feature.values())

    return drift_detected, {
        "drift_detected": drift_detected,
        "psi_per_feature": psi_per_feature,
        "psi_threshold": psi_threshold,
        "week_base_max": int(base_df["semana_num"].max()),
        "week_new": int(max_week),
        "n_base": len(base_df),
        "n_new": len(new_df),
    }
