import numpy as np
import pandas as pd

from gdp_nowcast import preprocessing as pp


def test_adf_detects_stationarity_white_noise():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(size=200))
    assert pp.adf_test(s)["stationary"] is True


def test_adf_random_walk_not_stationary():
    rng = np.random.default_rng(1)
    s = pd.Series(np.cumsum(rng.normal(size=200)))
    assert pp.adf_test(s)["stationary"] is False


def test_differencer_inverts_to_levels():
    idx = pd.date_range("2020-01-01", periods=10, freq="QS")
    levels = pd.Series(np.arange(10, 20, dtype="float64"), index=idx)
    d = pp.Differencer(order=1)
    diffs = d.fit_transform(levels)
    rebuilt = d.invert(levels.iloc[0], diffs.values)
    assert np.allclose(rebuilt, levels.values[1:])


def test_make_stationary_returns_order():
    rng = np.random.default_rng(2)
    # random walk com drift: claramente não-estacionário
    rw = pd.Series(np.cumsum(0.5 + rng.normal(size=200)))
    out, d = pp.make_stationary(rw, max_diff=2)
    if not pp.adf_test(rw)["stationary"]:
        assert d >= 1
    assert pp.adf_test(out)["stationary"]
