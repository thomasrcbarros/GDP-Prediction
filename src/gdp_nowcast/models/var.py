"""Modelo VAR multivariado para nowcasting do PIB.

Usa a série-alvo (pib_growth) em conjunto com os indicadores antecedentes.
Cada coluna é diferenciada se não for estacionária (ADF); a previsão do alvo é
reconstruída para a escala original quando houver diferenciação.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

from ..preprocessing import adf_test
from .base import NowcastModel


class VarModel(NowcastModel):
    name = "var"

    def __init__(self, maxlags: int = 4, ic: str = "aic"):
        self.maxlags = maxlags
        self.ic = ic
        self.k_diff: dict[str, int] = {}
        self._anchors: dict[str, float] = {}
        self._result = None
        self._columns: list[str] = []
        self.selected_lag: int | None = None

    def fit(self, target: pd.Series, exog: pd.DataFrame | None = None) -> "VarModel":
        if exog is None or exog.empty:
            raise ValueError("VAR requer indicadores (exog).")
        df = pd.concat([target.rename("pib_growth"), exog], axis=1).dropna()
        self._columns = list(df.columns)

        transformed = {}
        for col in df.columns:
            series = df[col]
            if adf_test(series)["stationary"]:
                self.k_diff[col] = 0
                transformed[col] = series
            else:
                self.k_diff[col] = 1
                self._anchors[col] = float(series.iloc[-1])
                transformed[col] = series.diff()
        tdf = pd.DataFrame(transformed).dropna()

        maxlags = min(self.maxlags, max(1, len(tdf) // (len(self._columns) + 1) - 1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = VAR(tdf)
            self._result = model.fit(maxlags=maxlags, ic=self.ic)
        self.selected_lag = int(self._result.k_ar)
        self._last_obs = tdf.values[-self._result.k_ar :]
        return self

    def forecast(self, steps: int = 1) -> pd.Series:
        if self._result is None:
            raise RuntimeError("Modelo não treinado.")
        lag = max(self._result.k_ar, 1)
        fc = self._result.forecast(self._last_obs[-lag:], steps=steps)
        fc_df = pd.DataFrame(fc, columns=self._columns)
        target_fc = fc_df["pib_growth"].values
        if self.k_diff.get("pib_growth", 0) == 1:
            target_fc = self._anchors["pib_growth"] + np.cumsum(target_fc)
        return pd.Series(target_fc, name="pib_growth")

    def summary(self) -> str:
        return f"VAR(p={self.selected_lag}, vars={len(self._columns)})"
