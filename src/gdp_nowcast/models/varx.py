"""VARX model (conditional VAR) for GDP nowcasting.

The pure VAR (``var.py``) forecasts the indicators themselves, wasting the
nowcasting advantage. The VARX estimates the same joint system [pib + indicators],
but at forecast time it **conditions** GDP on the values of the current quarter's
indicators that **have already been published** (IBC-Br, PIM, PMS, PMC are
released before GDP).

Gaussian conditioning: given the one-step joint forecast
``mu = (mu_y, mu_w)`` and the residual covariance ``Sigma``, the GDP forecast
(y) conditioned on the observed indicators ``w_obs`` is

    y_cond = mu_y + Sigma_yw · Sigma_ww^{-1} · (w_obs - mu_w).

True exogenous regressors (e.g.: the COVID peak dummy 2020Q1-Q2) are NOT
endogenized nor conditioned: they enter the VAR's ``exog``. Their columns are
provided via ``exog_cols`` in the constructor.

When the contemporaneous indicators are not provided, it falls back to the
unconditional forecast (equivalent to the pure VAR).
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
        # columns treated as true exogenous (not endogenous/conditioned)
        self.exog_cols = list(exog_cols) if exog_cols else []
        self._exog_active: list[str] = []  # exogenous actually used (non-constant)
        self.k_diff: dict[str, int] = {}
        self._anchors: dict[str, float] = {}
        self._result = None
        self._columns: list[str] = []   # endogenous columns (pib + indicators)
        self.selected_lag: int | None = None

    def fit(self, target: pd.Series, exog: pd.DataFrame | None = None) -> "VarxModel":
        if exog is None or exog.empty:
            raise ValueError("VARX requires indicators.")
        # separate true exogenous (e.g.: COVID dummy) from the endogenous indicators
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

        # Drop exogenous columns that are constant in the training window (e.g.: the
        # COVID dummy in windows entirely prior to 2020): a zero-variance column makes
        # the exogenous regression singular and produces NaN forecasts.
        if exog_true:
            sub = df[exog_true].loc[tdf.index]
            exog_true = [c for c in exog_true if float(sub[c].std(ddof=0)) > 1e-12]
        self._exog_active = exog_true
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
            raise RuntimeError("Model not trained.")
        p = max(self._result.k_ar, 1)

        # future value of the exogenous variables actually used (COVID dummy = 0 in the future)
        z_future = None
        if self._exog_active:
            if exog_future is not None and all(c in exog_future.columns for c in self._exog_active):
                z_future = exog_future[self._exog_active].to_numpy(dtype="float64")
            else:
                z_future = np.zeros((steps, len(self._exog_active)))

        fc_kwargs = {"steps": steps}
        if z_future is not None:
            fc_kwargs["exog_future"] = z_future
        mu = self._result.forecast(self._tdf.values[-p:], **fc_kwargs)[0]  # 1 step
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
        """Gaussian conditioning of GDP on the observed indicators.

        ``obs`` carries the indicators in the SAME transformed space as the model
        (% Q/Q change), hence it undergoes the same differencing if applicable.
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
        extra = f", exog={len(self._exog_active)}" if self.exog_cols else ""
        return f"VARX(p={self.selected_lag}, vars={len(self._columns)}, cond{extra})"
