"""Interface comum dos modelos de nowcasting."""
from __future__ import annotations

import abc

import pandas as pd


class NowcastModel(abc.ABC):
    """Contrato comum: ``fit`` no histórico e ``forecast`` h passos à frente.

    Todos os modelos preveem a série-alvo ``pib_growth``. Modelos univariados
    ignoram os indicadores; o VAR os utiliza.
    """

    name: str = "base"

    @abc.abstractmethod
    def fit(self, target: pd.Series, exog: pd.DataFrame | None = None) -> "NowcastModel":
        ...

    @abc.abstractmethod
    def forecast(self, steps: int = 1, exog_future: pd.DataFrame | None = None) -> pd.Series:
        """Retorna previsão de ``steps`` passos para ``pib_growth``.

        ``exog_future`` traz os regressores exógenos do(s) período(s) previsto(s)
        — usado por modelos com intervenção (dummies de COVID) e pela bridge
        equation (indicadores contemporâneos já observados). Modelos que não
        usam exógenas ignoram o argumento.
        """
        ...

    def summary(self) -> str:
        return f"{self.name} model"
