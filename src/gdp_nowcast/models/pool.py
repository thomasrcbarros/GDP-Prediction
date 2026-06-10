"""Pool of bridges with equal weights (forecast combination).

Combines forecasts from several bridge equations — each component uses a
subset of regressors — by the simple arithmetic mean. The forecast combination
literature (Timmermann, 2006) shows that equal weights are a benchmark that is
hard to beat; here the pool also gains robustness to the unavailability of
indicators: components without data in the forecast quarter are simply excluded
from the mean.
"""
from __future__ import annotations

import pandas as pd

from .base import NowcastModel
from .bridge import BridgeModel


class PoolModel(NowcastModel):
    name = "pool"

    def __init__(self, components: dict[str, list[str]]):
        self.components = components
        self._models: dict[str, BridgeModel] = {}
        self._covid_cols: list[str] = []

    def fit(self, target: pd.Series, exog: pd.DataFrame | None = None) -> "PoolModel":
        if exog is None or exog.empty:
            raise ValueError("The pool of bridges requires indicators (exog).")
        self._covid_cols = [c for c in exog.columns if c.startswith("covid_")]
        self._models = {}
        for comp_name, cols in self.components.items():
            bridge = BridgeModel(use_lagged_target=True)
            bridge.fit(target, exog[cols + self._covid_cols])
            self._models[comp_name] = bridge
        return self

    def forecast(self, steps: int = 1, exog_future: pd.DataFrame | None = None) -> pd.Series:
        if not self._models:
            raise RuntimeError("Model not trained.")
        if exog_future is None or exog_future.empty:
            raise ValueError("The pool needs the contemporaneous indicators (exog_future).")
        preds = []
        for comp_name, bridge in self._models.items():
            cols = self.components[comp_name]
            if exog_future[cols].isna().any().any():
                continue
            preds.append(bridge.forecast(steps, exog_future[cols + self._covid_cols]))
        if not preds:
            raise ValueError("No bridge in the pool has indicators available.")
        mean = sum(p.values for p in preds) / len(preds)
        return pd.Series(mean, name="pib_growth")

    def summary(self) -> str:
        return f"Pool(equal-weight, {len(self._models)} bridges)"
