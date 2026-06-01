"""Bridge equation: o nowcasting de verdade.

Regride o crescimento do PIB do trimestre sobre os indicadores antecedentes
*contemporâneos* (IBC-Br, produção industrial, desocupação) — que já estão
publicados quando o nowcast é feito, antes da divulgação oficial do PIB.

Diferentemente do ARIMA/SARIMA (extrapolação univariada) e do VAR (que prevê os
próprios indicadores), a bridge usa a informação já observada do trimestre
corrente, que é exatamente a vantagem informacional do nowcasting. Por isso é o
modelo que deve superar o random walk.

Opcionalmente inclui a defasagem do PIB e dummies de COVID como regressores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .base import NowcastModel


class BridgeModel(NowcastModel):
    name = "bridge"

    def __init__(self, use_lagged_target: bool = True):
        self.use_lagged_target = use_lagged_target
        self._result = None
        self._cols: list[str] = []
        self._last_target: float | None = None

    def _design(self, target: pd.Series, exog: pd.DataFrame) -> pd.DataFrame:
        X = exog.copy()
        if self.use_lagged_target:
            X["pib_lag1"] = target.shift(1)
        return X

    def fit(self, target: pd.Series, exog: pd.DataFrame | None = None) -> "BridgeModel":
        if exog is None or exog.empty:
            raise ValueError("A bridge equation requer indicadores (exog).")
        y = target.rename("pib_growth")
        X = self._design(y, exog)
        df = pd.concat([y, X], axis=1).dropna()
        self._cols = list(X.columns)
        self._last_target = float(y.dropna().iloc[-1])
        Xc = sm.add_constant(df[self._cols], has_constant="add")
        self._result = sm.OLS(df["pib_growth"], Xc).fit()
        return self

    def forecast(self, steps: int = 1, exog_future: pd.DataFrame | None = None) -> pd.Series:
        if self._result is None:
            raise RuntimeError("Modelo não treinado.")
        if exog_future is None or exog_future.empty:
            raise ValueError(
                "A bridge precisa dos indicadores contemporâneos (exog_future)."
            )
        Xf = exog_future.copy()
        if self.use_lagged_target and "pib_lag1" not in Xf.columns:
            # a defasagem do PIB no período previsto é o último valor observado
            Xf["pib_lag1"] = self._last_target
        Xf = Xf[[c for c in self._cols]]
        Xf = sm.add_constant(Xf, has_constant="add")
        pred = self._result.predict(Xf)
        return pd.Series(np.asarray(pred), name="pib_growth")

    def summary(self) -> str:
        return f"Bridge(OLS, {len(self._cols)} regressores)"
