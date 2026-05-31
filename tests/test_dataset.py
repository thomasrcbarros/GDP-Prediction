import numpy as np
import pandas as pd

import config
from gdp_nowcast import dataset


def _monthly(start, periods, step=1.0, base=100.0):
    idx = pd.date_range(start, periods=periods, freq="MS")
    return pd.Series(base + step * np.arange(periods), index=idx)


def test_to_quarterly_mean_aligns_to_quarter_start():
    s = _monthly("2020-01-01", 6, step=1.0, base=10.0)  # 10..15
    q = dataset.to_quarterly(s, "mean")
    assert list(q.index) == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-04-01")]
    assert q.iloc[0] == np.mean([10, 11, 12])
    assert q.iloc[1] == np.mean([13, 14, 15])


def test_pib_growth_qoq():
    idx = pd.date_range("2020-01-01", periods=3, freq="QS")
    pib = pd.Series([100.0, 102.0, 102.0], index=idx)
    g = dataset.pib_growth(pib, "qoq")
    assert np.isclose(g.iloc[0], 2.0)
    assert np.isclose(g.iloc[1], 0.0)


def test_available_as_of_blocks_unpublished_data():
    # série trimestral; lag de publicação de 60 dias
    spec = config.SeriesSpec(
        name="x", source="bcb", code="0", freq="Q", publication_lag_days=60
    )
    idx = pd.date_range("2020-01-01", periods=4, freq="QS")
    s = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
    # Em 2020-08-01: Q1 (fim 03-31 + 60d = 05-30) publicado; Q2 (06-30+60=08-29) ainda não
    avail = dataset.available_as_of(s, spec, pd.Timestamp("2020-08-01"))
    assert list(avail.index) == [pd.Timestamp("2020-01-01")]


def test_build_dataset_respects_lag_no_lookahead():
    raw = {
        "pib": pd.Series(
            np.linspace(100, 110, 12),
            index=pd.date_range("2019-01-01", periods=12, freq="QS"),
        ),
    }
    for spec in config.INDICATORS:
        raw[spec.name] = _monthly("2019-01-01", 36)

    as_of = pd.Timestamp("2021-02-15")
    df = dataset.build_dataset(
        raw, as_of=as_of, respect_publication_lag=True
    )
    # nenhuma data de publicação pode ultrapassar as_of
    for spec in config.ALL_SERIES:
        if spec.name == "pib":
            col = df["pib_growth"]
        else:
            col = df[spec.name]
        col = col.dropna()
        for q in col.index:
            pub = q + pd.offsets.QuarterEnd(0) + pd.Timedelta(days=spec.publication_lag_days)
            assert pub <= as_of, f"vazamento em {spec.name} no trimestre {q}"
