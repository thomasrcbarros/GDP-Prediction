"""Modelo SARIMA (sazonal) com seleção parcial de ordem por AIC."""
from __future__ import annotations

import itertools
import warnings

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

import config

from .base import NowcastModel


class SarimaModel(NowcastModel):
    name = "sarima"

    def __init__(
        self,
        max_p: int = 2,
        max_d: int = 1,
        max_q: int = 2,
        seasonal_period: int = config.SEASONAL_PERIOD,
    ):
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.s = seasonal_period
        self.order = None
        self.seasonal_order = None
        self._result = None

    def _select_order(self, y: pd.Series):
        best_aic = float("inf")
        best = ((1, 1, 1), (0, 1, 1, self.s))
        pdq = list(
            itertools.product(
                range(self.max_p + 1), range(self.max_d + 1), range(self.max_q + 1)
            )
        )
        seasonal = [(P, D, Q, self.s) for P in (0, 1) for D in (0, 1) for Q in (0, 1)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for order in pdq:
                for sorder in seasonal:
                    try:
                        res = SARIMAX(
                            y,
                            order=order,
                            seasonal_order=sorder,
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        ).fit(disp=False)
                    except Exception:  # noqa: BLE001
                        continue
                    if res.aic < best_aic:
                        best_aic, best = res.aic, (order, sorder)
        return best

    def fit(self, target: pd.Series, exog: pd.DataFrame | None = None) -> "SarimaModel":
        y = target.dropna()
        self.order, self.seasonal_order = self._select_order(y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._result = SARIMAX(
                y,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
        return self

    def forecast(self, steps: int = 1) -> pd.Series:
        if self._result is None:
            raise RuntimeError("Modelo não treinado.")
        return self._result.forecast(steps=steps)

    def summary(self) -> str:
        return f"SARIMA{self.order}x{self.seasonal_order}"
