import numpy as np
import pandas as pd

from gdp_nowcast import backtest as bt
from gdp_nowcast import dataset
from gdp_nowcast.models import ArimaModel, BridgeModel, SarimaModel, VarModel, VarxModel


def _series(n=60, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2005-01-01", periods=n, freq="QS")
    # stationary AR(1) around 0.5
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.5 + 0.4 * y[t - 1] + rng.normal(scale=0.3)
    return pd.Series(y, index=idx, name="pib_growth")


def test_arima_fits_and_forecasts_one_step():
    y = _series()
    m = ArimaModel(max_p=2, max_d=1, max_q=2).fit(y)
    fc = m.forecast(1)
    assert len(fc) == 1
    assert np.isfinite(fc.iloc[0])


def test_sarima_fits_and_forecasts():
    y = _series()
    m = SarimaModel(max_p=1, max_d=1, max_q=1).fit(y)
    fc = m.forecast(1)
    assert np.isfinite(fc.iloc[0])


def test_var_uses_exog_and_forecasts():
    y = _series()
    rng = np.random.default_rng(7)
    n = len(y)
    # indicators correlated with y but with independent noise (not collinear)
    exog = pd.DataFrame(
        {
            "ibcbr": 0.6 * y.values + rng.normal(scale=0.5, size=n),
            "pim": 0.4 * y.shift(1).bfill().values + rng.normal(scale=0.5, size=n),
            "desocupacao": 12 - 0.3 * y.values + rng.normal(scale=0.5, size=n),
        },
        index=y.index,
    )
    m = VarModel(maxlags=3).fit(y, exog)
    fc = m.forecast(1)
    assert np.isfinite(fc.iloc[0])
    assert m.selected_lag is not None


def test_varx_accepts_exog_dummy_column():
    y = _series()
    rng = np.random.default_rng(11)
    n = len(y)
    exog = pd.DataFrame(
        {
            "pim": 0.5 * y.values + rng.normal(scale=0.4, size=n),
            "covid_peak": dataset.covid_peak_dummy(y.index)["covid_peak"].values,
        },
        index=y.index,
    )
    m = VarxModel(maxlags=2, exog_cols=["covid_peak"]).fit(y, exog)
    # covid_peak is exogenous -> does not become an endogenous column of the system
    assert "covid_peak" not in m._columns
    assert "pim" in m._columns
    fut = pd.DataFrame({"pim": [1.0], "covid_peak": [0.0]})
    fc = m.forecast(1, exog_future=fut)
    assert np.isfinite(fc.iloc[0])


def test_rmse_period_splits_pre_post():
    idx = pd.date_range("2018-01-01", periods=12, freq="QS")
    actual = pd.Series(np.zeros(12), index=idx, name="actual")
    pred = pd.Series(np.ones(12), index=idx, name="pred")  # constant error = 1
    res = bt.BacktestResult("x", pred, actual, bt.compute_metrics(actual.values, pred.values))
    assert abs(bt.rmse_period(res, end="2020-01-01") - 1.0) < 1e-9
    assert abs(bt.rmse_period(res, start="2020-01-01") - 1.0) < 1e-9
    # empty window -> nan
    assert np.isnan(bt.rmse_period(res, start="2030-01-01"))


def test_rolling_backtest_train_window_limits_history():
    y = _series(n=40)
    res = bt.rolling_backtest(
        y, None, lambda: ArimaModel(max_p=1, max_d=1, max_q=1),
        min_train=20, train_window=12,
    )
    assert len(res.predictions) > 0
    assert np.isfinite(res.metrics["rmse"])


def test_varx_conditions_on_contemporaneous_indicators():
    y = _series()
    rng = np.random.default_rng(9)
    n = len(y)
    exog = pd.DataFrame(
        {
            "ibcbr": 0.7 * y.values + rng.normal(scale=0.3, size=n),
            "pim": 0.4 * y.values + rng.normal(scale=0.5, size=n),
        },
        index=y.index,
    )
    m = VarxModel(maxlags=3).fit(y, exog)
    # conditioning should differ from the unconditional forecast
    fut = pd.DataFrame({"ibcbr": [2.0], "pim": [1.5]})
    cond = m.forecast(1, exog_future=fut).iloc[0]
    uncond = m.forecast(1, exog_future=None).iloc[0]
    assert np.isfinite(cond) and np.isfinite(uncond)
    assert abs(cond - uncond) > 1e-9


def test_bridge_uses_contemporaneous_exog():
    y = _series()
    rng = np.random.default_rng(3)
    # indicator strongly tied to the contemporaneous target
    exog = pd.DataFrame(
        {"ibcbr": y.values + rng.normal(scale=0.2, size=len(y))}, index=y.index
    )
    m = BridgeModel(use_lagged_target=False).fit(y, exog)
    # forecast using the contemporaneous indicator of the next period
    fut = pd.DataFrame({"ibcbr": [1.0]})
    fc = m.forecast(1, exog_future=fut)
    assert np.isfinite(fc.iloc[0])


def test_bridge_requires_exog_future():
    y = _series()
    exog = pd.DataFrame({"ibcbr": y.values}, index=y.index)
    m = BridgeModel(use_lagged_target=False).fit(y, exog)
    try:
        m.forecast(1, exog_future=None)
        assert False, "should require exog_future"
    except ValueError:
        pass


def test_arima_exog_intervention_runs():
    y = _series()
    covid = dataset.covid_dummies(y.index)  # all zero outside 2020 -> they are dropped
    m = ArimaModel(max_p=1, max_d=1, max_q=1).fit(y, covid)
    fc = m.forecast(1, exog_future=covid.iloc[[0]])
    assert np.isfinite(fc.iloc[0])


def test_covid_dummies_flag_only_2020():
    idx = pd.date_range("2019-01-01", periods=12, freq="QS")
    d = dataset.covid_dummies(idx)
    assert d.loc["2020-04-01", "covid_2020q2"] == 1.0
    assert d.loc["2019-01-01"].sum() == 0.0
    assert d.shape[1] == 4


def test_metrics_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0])
    assert bt.rmse(y, y) == 0.0
    assert bt.mae(y, y) == 0.0


def test_rolling_backtest_produces_predictions():
    y = _series(n=40)
    res = bt.rolling_backtest(y, None, lambda: ArimaModel(max_p=1, max_d=1, max_q=1), min_train=20)
    assert len(res.predictions) > 0
    assert "rmse" in res.metrics
    assert np.isfinite(res.metrics["rmse"])
