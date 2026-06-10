"""Utilities for handling exogenous regressors in the models."""
from __future__ import annotations

import numpy as np
import pandas as pd


def drop_constant_columns(exog: pd.DataFrame | None) -> pd.DataFrame | None:
    """Remove constant columns (zero variance).

    Needed because, in backtest windows prior to 2020, the COVID dummies are
    all zero — constant columns break the estimation (singular matrix). Returns
    ``None`` if no column remains.
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
    """Align the forecast exog to the columns used in training.

    Keeps only ``cols`` (in the training order); missing columns become 0.
    Returns ``None`` if the model was trained without exogenous variables.
    """
    if not cols:
        return None
    if exog_future is None:
        return pd.DataFrame(np.zeros((steps, len(cols))), columns=cols)
    out = pd.DataFrame(index=exog_future.index)
    for c in cols:
        out[c] = exog_future[c] if c in exog_future.columns else 0.0
    return out
