"""Utilidades para tratar regressores exógenos nos modelos."""
from __future__ import annotations

import numpy as np
import pandas as pd


def drop_constant_columns(exog: pd.DataFrame | None) -> pd.DataFrame | None:
    """Remove colunas constantes (variância zero).

    Necessário porque, em janelas de backtest anteriores a 2020, as dummies de
    COVID são todas zero — colunas constantes quebram a estimação (matriz
    singular). Retorna ``None`` se não sobrar nenhuma coluna.
    """
    if exog is None:
        return None
    keep = [c for c in exog.columns if exog[c].std(skipna=True) > 0]
    if not keep:
        return None
    return exog[keep]


def align_exog(
    exog_future: pd.DataFrame | None, cols: list[str], steps: int
) -> pd.DataFrame | None:
    """Alinha o exog de previsão às colunas usadas no treino.

    Mantém apenas ``cols`` (na ordem do treino); colunas ausentes viram 0.
    Retorna ``None`` se o modelo foi treinado sem exógenas.
    """
    if not cols:
        return None
    if exog_future is None:
        return pd.DataFrame(np.zeros((steps, len(cols))), columns=cols)
    out = pd.DataFrame(index=exog_future.index)
    for c in cols:
        out[c] = exog_future[c] if c in exog_future.columns else 0.0
    return out
