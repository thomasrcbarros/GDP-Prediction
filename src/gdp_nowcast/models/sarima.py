"""SARIMA model (seasonal) with partial order selection by AIC.

Accepts optional exogenous regressors (e.g.: COVID intervention dummies),
passed to statsmodels' SARIMAX.
"""
from __future__ import annotations

import itertools
import warnings

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

import config

from .base import NowcastModel
from ._exog import align_exog, drop_constant_columns


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
        self._exog_cols: list[str] = []

    def _select_order(self, y: pd.Series, exog):
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
                            exog=exog,
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
        exog = drop_constant_columns(exog)
        if exog is not None:
            exog = exog.loc[y.index]
        self._exog_cols = list(exog.columns) if exog is not None else []
        self.order, self.seasonal_order = self._select_order(y, exog)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._result = SARIMAX(
                y,
                exog=exog,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
        return self

    def forecast(self, steps: int = 1, exog_future: pd.DataFrame | None = None) -> pd.Series:
        if self._result is None:
            raise RuntimeError("Model not trained.")
        exog_future = align_exog(exog_future, self._exog_cols, steps)
        return self._result.forecast(steps=steps, exog=exog_future)

    def summary(self) -> str:
        extra = f"+{len(self._exog_cols)} exog" if self._exog_cols else ""
        return f"SARIMA{self.order}x{self.seasonal_order}{extra}"
