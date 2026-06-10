"""Rolling-origin backtesting and look-ahead bias evaluation.

Two regimes are compared:

- **Realistic (out-of-sample)**: expanding window; the model is re-trained each
  quarter using *only* information prior to the forecast origin and forecasts
  the next quarter (one-step-ahead). This is how a real nowcast would
  operate.
- **Naive (look-ahead)**: the model is fit a single time over the entire sample
  (including future quarters) and the in-sample forecast is read for each
  quarter. This leaks future information and artificially inflates accuracy.

The difference between the two errors quantifies the look-ahead bias.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config


def render_progress(label: str, done: int, total: int, t0: float, width: int = 24) -> None:
    """Draws a single-line progress bar on stderr (no dependencies).

    Uses a carriage return (``\\r``) to update the same line; on completion
    (``done == total``) it emits the final line break.
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
        f"{frac*100:5.1f}%  {elapsed:4.0f}s elapsed  ETA {eta:4.0f}s"
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
    """Realistic one-step-ahead backtest.

    By default the training window is **expanding** (uses the whole history up to
    the origin). If ``train_window`` (number of quarters) is given, it uses a
    fixed-size **rolling window** — e.g.: ``train_window=20`` ~ 5 years —,
    discarding old observations. This allows comparing "full sample" vs "rolling
    window" estimation.

    ``score_start`` defines from which date the forecasts enter the evaluation
    (common test window across variants). ``progress_label`` draws the progress
    bar on stderr.
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
        # exogenous of the forecast quarter: COVID dummies (known) and/or
        # contemporaneous indicators already published (case of the bridge equation).
        fut_x = exog.iloc[[i]] if exog is not None else None
        try:
            model = model_factory()
            model.fit(train_y, train_x)
            yhat = float(model.forecast(steps=1, exog_future=fut_x).iloc[0])
        except Exception:  # noqa: BLE001 - model may not converge in some window
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
    """Look-ahead backtest: fits on the entire sample and reads in-sample.

    Demonstrates the artificial optimization of the metrics when future
    information leaks into training (order and parameter selection uses the whole series).
    """
    target = target.dropna()
    try:
        model = model_factory()
        model.fit(target, exog)
        # in-sample one-step-ahead forecast via predict of the underlying result
        result = getattr(model, "_result", None)
        if result is not None and hasattr(result, "predict"):
            fitted = result.predict(start=min_train, end=len(target) - 1)
            fitted = np.asarray(fitted, dtype="float64")
        else:
            raise AttributeError
    except Exception:  # noqa: BLE001 - fallback: repeats last training value
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
    """Baseline: forecast = last observed value (random walk)."""
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
    """Baseline: forecast = historical (expanding) mean up to the previous period.

    For quarterly GDP growth (mean-reverting) it is usually a stronger baseline
    than the random walk.
    """
    target = target.dropna()
    preds = _slice_score(target.expanding().mean().shift(1), min_train, score_start)
    actuals = _slice_score(target, min_train, score_start)
    return BacktestResult(
        model_name="mean",
        predictions=preds,
        actuals=actuals,
        metrics=compute_metrics(actuals.values, preds.values),
    )


def rmse_excluding_years(result: BacktestResult, years: list[int]) -> float:
    """RMSE recomputed discarding quarters of the indicated years (e.g. COVID)."""
    mask = ~result.actuals.index.year.isin(years)
    if not mask.any():
        return float("nan")
    return rmse(result.actuals[mask].values, result.predictions[mask].values)


def rmse_period(result: BacktestResult, start=None, end=None) -> float:
    """RMSE restricted to an interval [start, end) of forecast dates.

    Useful for comparing pre-2020 vs post-2020 performance separately.
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
