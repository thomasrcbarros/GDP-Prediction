"""Backtesting rolling-origin e avaliação de look-ahead bias.

Dois regimes são comparados:

- **Realista (out-of-sample)**: janela expansível; o modelo é re-treinado a
  cada trimestre usando *apenas* informação anterior à origem da previsão e
  prevê o trimestre seguinte (one-step-ahead). É assim que um nowcast real
  operaria.
- **Ingênuo (look-ahead)**: o modelo é ajustado uma única vez sobre a amostra
  inteira (inclusive trimestres futuros) e lê-se a previsão in-sample para cada
  trimestre. Isso vaza informação do futuro e infla artificialmente a acurácia.

A diferença entre os dois erros quantifica o look-ahead bias.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def bias(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "n": int(len(y_true)),
    }


@dataclass
class BacktestResult:
    model_name: str
    predictions: pd.Series
    actuals: pd.Series
    metrics: dict


def rolling_backtest(
    target: pd.Series,
    exog: pd.DataFrame | None,
    model_factory,
    min_train: int = config.MIN_TRAIN_QUARTERS,
) -> BacktestResult:
    """Backtest realista one-step-ahead com janela expansível."""
    target = target.dropna()
    idx = target.index
    preds, actuals, dates = [], [], []

    for i in range(min_train, len(target)):
        train_y = target.iloc[:i]
        train_x = exog.iloc[:i] if exog is not None else None
        try:
            model = model_factory()
            model.fit(train_y, train_x)
            yhat = float(model.forecast(steps=1).iloc[0])
        except Exception:  # noqa: BLE001 - modelo pode não convergir em alguma janela
            continue
        preds.append(yhat)
        actuals.append(float(target.iloc[i]))
        dates.append(idx[i])

    pred_s = pd.Series(preds, index=pd.DatetimeIndex(dates), name="pred")
    act_s = pd.Series(actuals, index=pd.DatetimeIndex(dates), name="actual")
    model = model_factory()
    return BacktestResult(
        model_name=getattr(model, "name", "model"),
        predictions=pred_s,
        actuals=act_s,
        metrics=compute_metrics(act_s, pred_s),
    )


def naive_lookahead_backtest(
    target: pd.Series,
    exog: pd.DataFrame | None,
    model_factory,
    min_train: int = config.MIN_TRAIN_QUARTERS,
) -> BacktestResult:
    """Backtest com look-ahead: ajusta na amostra inteira e lê in-sample.

    Demonstra a otimização artificial das métricas quando informação futura
    vaza para o treino (seleção de ordem e parâmetros usa toda a série).
    """
    target = target.dropna()
    try:
        model = model_factory()
        model.fit(target, exog)
        # previsão in-sample one-step-ahead via predict do resultado subjacente
        result = getattr(model, "_result", None)
        if result is not None and hasattr(result, "predict"):
            fitted = result.predict(start=min_train, end=len(target) - 1)
            fitted = np.asarray(fitted, dtype="float64")
        else:
            raise AttributeError
    except Exception:  # noqa: BLE001 - fallback: repete último valor de treino
        fitted = target.iloc[min_train:].shift(1).bfill().values

    idx = target.index[min_train:]
    n = min(len(fitted), len(idx))
    pred_s = pd.Series(np.asarray(fitted)[:n], index=idx[:n], name="pred")
    act_s = target.iloc[min_train : min_train + n]
    act_s.name = "actual"
    return BacktestResult(
        model_name=getattr(model_factory(), "name", "model"),
        predictions=pred_s,
        actuals=act_s,
        metrics=compute_metrics(act_s.values, pred_s.values),
    )


def random_walk_baseline(
    target: pd.Series, min_train: int = config.MIN_TRAIN_QUARTERS
) -> BacktestResult:
    """Baseline: previsão = último valor observado (random walk)."""
    target = target.dropna()
    preds = target.shift(1).iloc[min_train:]
    actuals = target.iloc[min_train:]
    return BacktestResult(
        model_name="random_walk",
        predictions=preds,
        actuals=actuals,
        metrics=compute_metrics(actuals.values, preds.values),
    )
