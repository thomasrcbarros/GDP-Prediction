"""Modelo ARIMA com seleção de ordem (p,d,q) por AIC."""
from __future__ import annotations

import itertools
import warnings

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from .base import NowcastModel


class ArimaModel(NowcastModel):
    name = "arima"

    def __init__(self, max_p: int = 3, max_d: int = 2, max_q: int = 3):
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.order: tuple[int, int, int] | None = None
        self._result = None

    def _select_order(self, y: pd.Series) -> tuple[int, int, int]:
        best_aic = float("inf")
        best_order = (1, 1, 1)
        grid = itertools.product(
            range(self.max_p + 1), range(self.max_d + 1), range(self.max_q + 1)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for order in grid:
                if order == (0, 0, 0):
                    continue
                try:
                    res = ARIMA(y, order=order).fit()
                except Exception:  # noqa: BLE001 - ordens inválidas/não convergem
                    continue
                if res.aic < best_aic:
                    best_aic, best_order = res.aic, order
        return best_order

    def fit(self, target: pd.Series, exog: pd.DataFrame | None = None) -> "ArimaModel":
        y = target.dropna()
        self.order = self._select_order(y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._result = ARIMA(y, order=self.order).fit()
        return self

    def forecast(self, steps: int = 1) -> pd.Series:
        if self._result is None:
            raise RuntimeError("Modelo não treinado.")
        return self._result.forecast(steps=steps)

    def summary(self) -> str:
        return f"ARIMA{self.order}"
