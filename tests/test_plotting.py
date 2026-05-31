import os

import numpy as np
import pandas as pd

from gdp_nowcast import plotting
from gdp_nowcast.backtest import BacktestResult, compute_metrics


def _result(name, n=12):
    idx = pd.date_range("2018-01-01", periods=n, freq="QS")
    actual = pd.Series(np.linspace(-1, 2, n), index=idx, name="actual")
    pred = actual + np.random.default_rng(0).normal(scale=0.3, size=n)
    pred.name = "pred"
    return BacktestResult(name, pred, actual, compute_metrics(actual.values, pred.values))


def test_plot_comparison_creates_file(tmp_path):
    results = {m: _result(m) for m in ("arima", "sarima", "var")}
    rows = []
    for m, r in results.items():
        rows.append({"model": m, "regime": "realista", **r.metrics})
        rows.append({"model": m, "regime": "look-ahead", **r.metrics})
    table = pd.DataFrame(rows)
    nowcasts = {"arima": 0.6, "sarima": 0.5, "var": 1.0}

    out = os.path.join(tmp_path, "comp.png")
    path = plotting.plot_comparison(results, table, out, nowcasts=nowcasts)
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0
