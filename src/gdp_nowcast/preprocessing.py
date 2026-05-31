"""Pré-processamento: estacionariedade e transformações reversíveis."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def adf_test(series: pd.Series, signif: float = 0.05) -> dict:
    """Teste Augmented Dickey-Fuller de estacionariedade.

    Returns dict com estatística, p-valor e flag ``stationary``.
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
    """Diferenciação reversível de 1ª ordem.

    Guarda o último valor original para reconstruir a série em nível a partir
    das previsões diferenciadas.
    """

    order: int = 1
    _anchor: float | None = None

    def fit_transform(self, series: pd.Series) -> pd.Series:
        if self.order == 0:
            return series.copy()
        self._anchor = float(series.iloc[-1])
        return series.diff(self.order).dropna()

    def invert(self, last_level: float, diffs) -> np.ndarray:
        """Reconstrói níveis a partir de um nível inicial e das diferenças."""
        diffs = np.asarray(diffs, dtype="float64")
        return last_level + np.cumsum(diffs)


def make_stationary(series: pd.Series, max_diff: int = 2) -> tuple[pd.Series, int]:
    """Diferencia até a série passar no ADF (ou atingir ``max_diff``).

    Returns (série transformada, ordem de diferenciação aplicada).
    """
    s = series.dropna()
    for d in range(max_diff + 1):
        if d > 0:
            s = s.diff().dropna()
        if adf_test(s)["stationary"]:
            return s, d
    return s, max_diff
