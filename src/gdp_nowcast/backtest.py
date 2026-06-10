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

import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config


def render_progress(label: str, done: int, total: int, t0: float, width: int = 24) -> None:
    """Desenha uma barra de progresso de uma linha no stderr (sem dependências).

    Usa retorno de carro (``\\r``) para atualizar a mesma linha; ao concluir
    (``done == total``) emite a quebra de linha final.
    """
    if total <= 0:
        return
    frac = done / total
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.time() - t0
    eta = (elapsed / done * (total - done)) if done else 0.0
    sys.stderr.write(
        f"\r  {label:<22s} [{bar}] {done:>3d}/{total:<3d} "
        f"{frac*100:5.1f}%  {elapsed:4.0f}s decorrido  ETA {eta:4.0f}s"
    )
    sys.stderr.flush()
    if done >= total:
        sys.stderr.write("\n")
        sys.stderr.flush()


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
    progress_label: str | None = None,
    score_start: pd.Timestamp | None = None,
    train_window: int | None = None,
) -> BacktestResult:
    """Backtest realista one-step-ahead.

    Por padrão a janela de treino é **expansível** (usa todo o histórico até a
    origem). Se ``train_window`` (nº de trimestres) for dado, usa uma **janela
    rolante** de tamanho fixo — ex.: ``train_window=20`` ≈ 5 anos —, descartando
    observações antigas. Isso permite comparar "amostra completa" vs "janela
    rolante" de estimação.

    ``score_start`` define a partir de qual data as previsões entram na avaliação
    (janela de teste comum entre variantes). ``progress_label`` desenha a barra
    de progresso no stderr.
    """
    target = target.dropna()
    if exog is not None:
        exog = exog.reindex(target.index)
    idx = target.index
    preds, actuals, dates = [], [], []

    steps_idx = [
        i
        for i in range(min_train, len(target))
        if score_start is None or idx[i] >= score_start
    ]
    total = len(steps_idx)
    t0 = time.time()
    for step, i in enumerate(steps_idx, start=1):
        lo = max(0, i - train_window) if train_window else 0
        train_y = target.iloc[lo:i]
        train_x = exog.iloc[lo:i] if exog is not None else None
        # exógenas do trimestre previsto: dummies de COVID (conhecidas) e/ou
        # indicadores contemporâneos já publicados (caso da bridge equation).
        fut_x = exog.iloc[[i]] if exog is not None else None
        try:
            model = model_factory()
            model.fit(train_y, train_x)
            yhat = float(model.forecast(steps=1, exog_future=fut_x).iloc[0])
        except Exception:  # noqa: BLE001 - modelo pode não convergir em alguma janela
            if progress_label is not None:
                render_progress(progress_label, step, total, t0)
            continue
        preds.append(yhat)
        actuals.append(float(target.iloc[i]))
        dates.append(idx[i])
        if progress_label is not None:
            render_progress(progress_label, step, total, t0)

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


def _slice_score(series: pd.Series, min_train: int, score_start):
    out = series.iloc[min_train:]
    if score_start is not None:
        out = out[out.index >= score_start]
    return out


def random_walk_baseline(
    target: pd.Series,
    min_train: int = config.MIN_TRAIN_QUARTERS,
    score_start: pd.Timestamp | None = None,
) -> BacktestResult:
    """Baseline: previsão = último valor observado (random walk)."""
    target = target.dropna()
    preds = _slice_score(target.shift(1), min_train, score_start)
    actuals = _slice_score(target, min_train, score_start)
    return BacktestResult(
        model_name="random_walk",
        predictions=preds,
        actuals=actuals,
        metrics=compute_metrics(actuals.values, preds.values),
    )


def mean_baseline(
    target: pd.Series,
    min_train: int = config.MIN_TRAIN_QUARTERS,
    score_start: pd.Timestamp | None = None,
) -> BacktestResult:
    """Baseline: previsão = média histórica (expansível) até o período anterior.

    Para o crescimento trimestral do PIB (mean-reverting) costuma ser um
    baseline mais forte que o random walk.
    """
    target = target.dropna()
    preds = _slice_score(target.expanding().mean().shift(1), min_train, score_start)
    actuals = _slice_score(target, min_train, score_start)
    return BacktestResult(
        model_name="media",
        predictions=preds,
        actuals=actuals,
        metrics=compute_metrics(actuals.values, preds.values),
    )


def rmse_excluding_years(result: BacktestResult, years: list[int]) -> float:
    """RMSE recomputado descartando trimestres dos anos indicados (ex.: COVID)."""
    mask = ~result.actuals.index.year.isin(years)
    if not mask.any():
        return float("nan")
    return rmse(result.actuals[mask].values, result.predictions[mask].values)


def rmse_period(result: BacktestResult, start=None, end=None) -> float:
    """RMSE restrito a um intervalo [start, end) de datas das previsões.

    Útil para comparar desempenho pré-2020 vs pós-2020 separadamente.
    """
    idx = result.actuals.index
    mask = pd.Series(True, index=idx)
    if start is not None:
        mask &= idx >= pd.Timestamp(start)
    if end is not None:
        mask &= idx < pd.Timestamp(end)
    if not mask.any():
        return float("nan")
    return rmse(result.actuals[mask.values].values, result.predictions[mask.values].values)
