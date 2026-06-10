import numpy as np
import pandas as pd

import config
from gdp_nowcast import dataset


def _monthly(start, periods, step=1.0, base=100.0):
    idx = pd.date_range(start, periods=periods, freq="MS")
    return pd.Series(base + step * np.arange(periods), index=idx)


def test_covid_peak_dummy_marks_only_2020q1q2():
    idx = pd.date_range("2019-01-01", periods=10, freq="QS")
    d = dataset.covid_peak_dummy(idx)
    assert list(d.columns) == ["covid_peak"]
    assert d.loc["2020-01-01", "covid_peak"] == 1.0
    assert d.loc["2020-04-01", "covid_peak"] == 1.0
    assert d.loc["2020-07-01", "covid_peak"] == 0.0
    assert d.loc["2019-01-01", "covid_peak"] == 0.0
    assert d["covid_peak"].sum() == 2.0


def test_varx_e_composition():
    # endógenas: pib_growth (implícito) + ibcbr + spread
    assert config.VARX_E["endog"] == ["ibcbr", "spread"]
    # fator PCA de atividade + confiança + dummy COVID são as exógenas verdadeiras
    assert config.VARX_E["pca_inputs"] == ["pim", "pms", "pmc"]
    assert config.VARX_E_EXOG_COLS == ["fator_atividade", "confcons", "covid_peak"]
    # todas as séries-base do conjunto E precisam existir em SERIES
    for c in config.FEATURE_SETS["E"]:
        assert c in config.SERIES


def test_pca_first_factor_summarizes_correlated_inputs():
    idx = pd.date_range("2010-01-01", periods=24, freq="QS")
    base = np.linspace(-1, 1, 24)
    df = pd.DataFrame(
        {"a": base + 0.01, "b": 2 * base, "c": 0.5 * base}, index=idx
    )
    f = dataset.pca_first_factor(df, "fator")
    assert f.name == "fator"
    # fator deve correlacionar positivamente com a média dos insumos
    assert np.corrcoef(f.values, df.mean(axis=1).values)[0, 1] > 0.9


def test_to_quarterly_mean_aligns_to_quarter_start():
    s = _monthly("2020-01-01", 6, step=1.0, base=10.0)  # 10..15
    q = dataset.to_quarterly(s, "mean")
    assert list(q.index) == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-04-01")]
    assert q.iloc[0] == np.mean([10, 11, 12])
    assert q.iloc[1] == np.mean([13, 14, 15])


def test_transform_indicator_growth_and_rate():
    s = _monthly("2020-01-01", 6, step=0.0, base=100.0)  # nível constante
    spec_g = config.SeriesSpec("x", "bcb", "0", "M", "mean", "growth")
    g = dataset.transform_indicator(s, spec_g)
    assert (g.abs() < 1e-9).all()  # nível constante -> variação 0
    spec_r = config.SeriesSpec("y", "bcb", "0", "M", "sum", "rate")
    r = dataset.transform_indicator(_monthly("2020-01-01", 3, step=0, base=1.0), spec_r)
    assert abs(r.iloc[0] - 3.0) < 1e-9  # soma trimestral de 1+1+1


def test_feature_sets_reference_known_series():
    # colunas derivadas (não são séries brutas; calculadas em _prepare)
    derived = {"ibcbr_own", *config.UMIDAS_COLS}
    for name, cols in config.FEATURE_SETS.items():
        for c in cols:
            if c in derived:
                continue
            assert c in config.SERIES, f"{c} do conjunto {name} não está em SERIES"


def test_stl_sa_growth_removes_seasonality():
    # nível com tendência + sazonalidade mensal forte
    idx = pd.date_range("2010-01-01", periods=120, freq="MS")
    trend = 100 + 0.3 * np.arange(120)
    seas = 5 * np.sin(2 * np.pi * (idx.month - 1) / 12)
    s = pd.Series(trend + seas, index=idx)
    g = dataset.stl_sa_growth(s, "qoq")
    assert g.index.freqstr in ("QS-JAN", "QS-OCT", "QS") or g.index.is_monotonic_increasing
    # crescimento dessaz deve ser suave (sem o pulso sazonal): desvio pequeno
    assert g.std() < 5.0
    assert np.isfinite(g).all()


def test_feature_set_a2_uses_two_seasonal_adjustments():
    assert config.FEATURE_SETS["A2"] == ["ibcbr", "ibcbr_own"]
    assert "ibcbr_nsa" in config.SERIES


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


def test_monthly_growth_features_pivots_months_into_quarter_columns():
    # 7 meses: jan-jul/2020; crescimento constante de 1 unidade sobre base 100
    s = _monthly("2020-01-01", 7, step=1.0, base=100.0)
    out = dataset.monthly_growth_features(s, "ibcbr")
    assert list(out.columns) == ["ibcbr_m1", "ibcbr_m2", "ibcbr_m3"]
    # m1 do Q1 é NaN (pct_change descarta o 1º mês)
    q1 = pd.Timestamp("2020-01-01")
    q2 = pd.Timestamp("2020-04-01")
    q3 = pd.Timestamp("2020-07-01")
    assert np.isnan(out.loc[q1, "ibcbr_m1"])
    assert abs(out.loc[q1, "ibcbr_m2"] - 1.0) < 1e-9   # 101/100 - 1
    assert abs(out.loc[q2, "ibcbr_m1"] - 100 * (103 / 102 - 1)) < 1e-9
    # Q3 incompleto: só m1 observado, m2/m3 NaN
    assert not np.isnan(out.loc[q3, "ibcbr_m1"])
    assert np.isnan(out.loc[q3, "ibcbr_m2"]) and np.isnan(out.loc[q3, "ibcbr_m3"])
