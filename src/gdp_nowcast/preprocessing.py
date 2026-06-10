"""Preprocessing: stationarity and reversible transformations."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def adf_test(series: pd.Series, signif: float = 0.05) -> dict:
    """Augmented Dickey-Fuller test for stationarity.

    Returns dict with statistic, p-value and the ``stationary`` flag.
    """
    s = series.dropna()
    if len(s) < 8:
        return {"stat": np.nan, "pvalue": np.nan, "stationary": False, "nobs": len(s)}
    stat, pvalue, _, nobs, _, _ = adfuller(s, autolag="AIC")
    return {
        "stat": float(stat),
        "pvalue": float(pvalue),
        "stationary": bool(pvalue < signif),
        "nobs": int(nobs),
    }


@dataclass
class Differencer:
    """Reversible 1st-order differencing.

    Stores the last original value to reconstruct the level series from the
    differenced forecasts.
    """

    order: int = 1
    _anchor: float | None = None

    def fit_transform(self, series: pd.Series) -> pd.Series:
        if self.order == 0:
            return series.copy()
        self._anchor = float(series.iloc[-1])
        return series.diff(self.order).dropna()

    def invert(self, last_level: float, diffs) -> np.ndarray:
        """Reconstructs levels from an initial level and the differences."""
        diffs = np.asarray(diffs, dtype="float64")
        return last_level + np.cumsum(diffs)


def make_stationary(series: pd.Series, max_diff: int = 2) -> tuple[pd.Series, int]:
    """Differences until the series passes the ADF test (or reaches ``max_diff``).

    Returns (transformed series, applied differencing order).
    """
    s = series.dropna()
    for d in range(max_diff + 1):
        if d > 0:
            s = s.diff().dropna()
        if adf_test(s)["stationary"]:
            return s, d
    return s, max_diff
