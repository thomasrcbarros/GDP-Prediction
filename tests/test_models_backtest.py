import numpy as np
import pandas as pd

from gdp_nowcast import backtest as bt
from gdp_nowcast import dataset
from gdp_nowcast.models import ArimaModel, BridgeModel, SarimaModel, VarModel


def _series(n=60, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2005-01-01", periods=n, freq="QS")
    # AR(1) estacionário em torno de 0.5
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
    # indicadores correlacionados com y mas com ruído independente (não colineares)
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


def test_bridge_uses_contemporaneous_exog():
    y = _series()
    rng = np.random.default_rng(3)
    # indicador fortemente ligado ao alvo contemporâneo
    exog = pd.DataFrame(
        {"ibcbr": y.values + rng.normal(scale=0.2, size=len(y))}, index=y.index
    )
    m = BridgeModel(use_lagged_target=False).fit(y, exog)
    # prever usando o indicador contemporâneo do próximo período
    fut = pd.DataFrame({"ibcbr": [1.0]})
    fc = m.forecast(1, exog_future=fut)
    assert np.isfinite(fc.iloc[0])


def test_bridge_requires_exog_future():
    y = _series()
    exog = pd.DataFrame({"ibcbr": y.values}, index=y.index)
    m = BridgeModel(use_lagged_target=False).fit(y, exog)
    try:
        m.forecast(1, exog_future=None)
        assert False, "deveria exigir exog_future"
    except ValueError:
        pass


def test_arima_exog_intervention_runs():
    y = _series()
    covid = dataset.covid_dummies(y.index)  # todas zero fora de 2020 -> são dropadas
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
