"""Pool de bridges com pesos iguais (forecast combination).

Combina previsões de várias bridge equations — cada componente usa um
subconjunto de regressores — pela média aritmética simples. A literatura de
combinação de previsões (Timmermann, 2006) mostra que pesos iguais são um
benchmark difícil de superar; aqui o pool também ganha robustez à
indisponibilidade de indicadores: componentes sem dados no trimestre previsto
são simplesmente excluídos da média.
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
            raise ValueError("O pool de bridges requer indicadores (exog).")
        self._covid_cols = [c for c in exog.columns if c.startswith("covid_")]
        self._models = {}
        for comp_name, cols in self.components.items():
            bridge = BridgeModel(use_lagged_target=True)
            bridge.fit(target, exog[cols + self._covid_cols])
            self._models[comp_name] = bridge
        return self

    def forecast(self, steps: int = 1, exog_future: pd.DataFrame | None = None) -> pd.Series:
        if not self._models:
            raise RuntimeError("Modelo não treinado.")
        if exog_future is None or exog_future.empty:
            raise ValueError("O pool precisa dos indicadores contemporâneos (exog_future).")
        preds = []
        for comp_name, bridge in self._models.items():
            cols = self.components[comp_name]
            if exog_future[cols].isna().any().any():
                continue
            preds.append(bridge.forecast(steps, exog_future[cols + self._covid_cols]))
        if not preds:
            raise ValueError("Nenhuma bridge do pool tem indicadores disponíveis.")
        mean = sum(p.values for p in preds) / len(preds)
        return pd.Series(mean, name="pib_growth")

    def summary(self) -> str:
        return f"Pool(equal-weight, {len(self._models)} bridges)"
