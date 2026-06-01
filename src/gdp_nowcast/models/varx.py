"""Modelo VARX (VAR condicional) para nowcasting do PIB.

O VAR puro (``var.py``) prevê os próprios indicadores, desperdiçando a vantagem
do nowcasting. O VARX estima o mesmo sistema conjunto [pib + indicadores], mas
na previsão **condiciona** o PIB nos valores dos indicadores do trimestre
corrente que **já foram publicados** (IBC-Br, PIM, PMS, PMC saem antes do PIB).

Condicionamento gaussiano: dada a previsão conjunta de um passo
``mu = (mu_y, mu_w)`` e a covariância dos resíduos ``Sigma``, a previsão do PIB
(y) condicionada nos indicadores observados ``w_obs`` é

    y_cond = mu_y + Sigma_yw · Sigma_ww^{-1} · (w_obs - mu_w).

Regressores exógenos verdadeiros (ex.: a dummy de pico da COVID 2020Q1-Q2) NÃO
são endogenizados nem condicionados: entram no ``exog`` do VAR. Suas colunas são
informadas via ``exog_cols`` no construtor.

Quando os indicadores contemporâneos não são fornecidos, recai na previsão
incondicional (equivalente ao VAR puro).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

from ..preprocessing import adf_test
from .base import NowcastModel


class VarxModel(NowcastModel):
    name = "varx"

    def __init__(self, maxlags: int = 4, ic: str = "aic", exog_cols: list[str] | None = None):
        self.maxlags = maxlags
        self.ic = ic
        # colunas tratadas como exógenas verdadeiras (não endógenas/condicionadas)
        self.exog_cols = list(exog_cols) if exog_cols else []
        self.k_diff: dict[str, int] = {}
        self._anchors: dict[str, float] = {}
        self._result = None
        self._columns: list[str] = []   # colunas endógenas (pib + indicadores)
        self.selected_lag: int | None = None

    def fit(self, target: pd.Series, exog: pd.DataFrame | None = None) -> "VarxModel":
        if exog is None or exog.empty:
            raise ValueError("VARX requer indicadores.")
        # separa exógenas verdadeiras (ex.: dummy COVID) dos indicadores endógenos
        exog_true = [c for c in self.exog_cols if c in exog.columns]
        endog_ind = [c for c in exog.columns if c not in exog_true]

        df = pd.concat([target.rename("pib_growth"), exog], axis=1).dropna()
        self._columns = ["pib_growth"] + endog_ind

        transformed = {}
        for col in self._columns:
            s = df[col]
            if adf_test(s)["stationary"]:
                self.k_diff[col] = 0
                transformed[col] = s
            else:
                self.k_diff[col] = 1
                self._anchors[col] = float(s.iloc[-1])
                transformed[col] = s.diff()
        tdf = pd.DataFrame(transformed).dropna()

        z = df[exog_true].loc[tdf.index] if exog_true else None

        maxlags = min(self.maxlags, max(1, len(tdf) // (len(self._columns) + 1) - 1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._result = VAR(tdf, exog=z).fit(maxlags=maxlags, ic=self.ic)
        self.selected_lag = int(self._result.k_ar)
        self._tdf = tdf
        return self

    def forecast(self, steps: int = 1, exog_future: pd.DataFrame | None = None) -> pd.Series:
        if self._result is None:
            raise RuntimeError("Modelo não treinado.")
        p = max(self._result.k_ar, 1)

        # valor futuro dos exógenos verdadeiros (dummy COVID = 0 no futuro)
        z_future = None
        if self.exog_cols:
            if exog_future is not None and all(c in exog_future.columns for c in self.exog_cols):
                z_future = exog_future[self.exog_cols].to_numpy(dtype="float64")
            else:
                z_future = np.zeros((steps, len(self.exog_cols)))

        fc_kwargs = {"steps": steps}
        if z_future is not None:
            fc_kwargs["exog_future"] = z_future
        mu = self._result.forecast(self._tdf.values[-p:], **fc_kwargs)[0]  # 1 passo
        mu = dict(zip(self._columns, mu))

        y_hat = mu["pib_growth"]
        w_names = [c for c in self._columns if c != "pib_growth"]
        if exog_future is not None and not exog_future.empty:
            row = exog_future.iloc[0]
            obs = {c: row[c] for c in w_names if c in row.index and pd.notna(row[c])}
            if obs:
                y_hat = self._condition(mu, obs)

        if self.k_diff.get("pib_growth", 0) == 1:
            y_hat = self._anchors["pib_growth"] + y_hat
        return pd.Series([y_hat], name="pib_growth")

    def _condition(self, mu: dict, obs: dict) -> float:
        """Condicionamento gaussiano do PIB nos indicadores observados.

        ``obs`` traz os indicadores no MESMO espaço transformado do modelo
        (variação % T/T), portanto sofre a mesma diferenciação se aplicável.
        """
        cols = self._columns
        sigma = np.asarray(self._result.sigma_u)
        yi = [cols.index("pib_growth")]
        w_cols = list(obs.keys())
        wi = [cols.index(c) for c in w_cols]

        w_obs = []
        for c in w_cols:
            v = obs[c]
            if self.k_diff.get(c, 0) == 1:
                v = v - self._anchors[c]
            w_obs.append(v)
        w_obs = np.asarray(w_obs, dtype="float64")

        syw = sigma[np.ix_(yi, wi)]
        sww = sigma[np.ix_(wi, wi)]
        mu_w = np.array([mu[c] for c in w_cols])
        adj = syw @ np.linalg.solve(sww, (w_obs - mu_w))
        return float(mu["pib_growth"] + adj[0])

    def summary(self) -> str:
        extra = f", exog={len(self.exog_cols)}" if self.exog_cols else ""
        return f"VARX(p={self.selected_lag}, vars={len(self._columns)}, cond{extra})"
