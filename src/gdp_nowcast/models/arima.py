"""ARIMA model with order selection (p,d,q) by AIC.

Accepts optional exogenous regressors (e.g.: COVID intervention dummies),
which are passed to statsmodels' ARIMAX.
"""
from __future__ import annotations

import itertools
import warnings

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from .base import NowcastModel
from ._exog import align_exog, drop_constant_columns


class ArimaModel(NowcastModel):
    name = "arima"

    def __init__(self, max_p: int = 3, max_d: int = 2, max_q: int = 3):
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.order: tuple[int, int, int] | None = None
        self._result = None
        self._exog_cols: list[str] = []

    def _select_order(self, y: pd.Series, exog) -> tuple[int, int, int]:
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
                    res = ARIMA(y, order=order, exog=exog).fit()
                except Exception:  # noqa: BLE001 - invalid orders/do not converge
                    continue
                if res.aic < best_aic:
                    best_aic, best_order = res.aic, order
        return best_order

    def fit(self, target: pd.Series, exog: pd.DataFrame | None = None) -> "ArimaModel":
        y = target.dropna()
        exog = drop_constant_columns(exog)
        if exog is not None:
            exog = exog.loc[y.index]
        self._exog_cols = list(exog.columns) if exog is not None else []
        self.order = self._select_order(y, exog)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._result = ARIMA(y, order=self.order, exog=exog).fit()
        return self

    def forecast(self, steps: int = 1, exog_future: pd.DataFrame | None = None) -> pd.Series:
        if self._result is None:
            raise RuntimeError("Model not trained.")
        exog_future = align_exog(exog_future, self._exog_cols, steps)
        return self._result.forecast(steps=steps, exog=exog_future)

    def summary(self) -> str:
        extra = f"+{len(self._exog_cols)} exog" if self._exog_cols else ""
        return f"ARIMA{self.order}{extra}"
