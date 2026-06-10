"""Common interface for the nowcasting models."""
from __future__ import annotations

import abc

import pandas as pd


class NowcastModel(abc.ABC):
    """Common contract: ``fit`` on history and ``forecast`` h steps ahead.

    All models forecast the target series ``pib_growth``. Univariate models
    ignore the indicators; the VAR uses them.
    """

    name: str = "base"

    @abc.abstractmethod
    def fit(self, target: pd.Series, exog: pd.DataFrame | None = None) -> "NowcastModel":
        ...

    @abc.abstractmethod
    def forecast(self, steps: int = 1, exog_future: pd.DataFrame | None = None) -> pd.Series:
        """Returns a forecast of ``steps`` steps for ``pib_growth``.

        ``exog_future`` carries the exogenous regressors of the forecast
        period(s) — used by models with intervention (COVID dummies) and by the
        bridge equation (contemporaneous indicators already observed). Models
        that do not use exogenous variables ignore the argument.
        """
        ...

    def summary(self) -> str:
        return f"{self.name} model"
