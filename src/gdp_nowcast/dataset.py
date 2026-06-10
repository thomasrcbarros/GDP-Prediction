"""Construction of the aligned quarterly dataset for nowcasting.

Responsible for:
- aggregating monthly indicators to quarterly frequency;
- building the target variable (GDP growth);
- simulating the real data availability at a reference date
  (publication lags) -- the central piece of the look-ahead bias evaluation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

_AGG_FUNCS = {"mean": "mean", "sum": "sum", "last": "last"}


def to_quarterly(series: pd.Series, agg: str) -> pd.Series:
    """Aggregates a series (monthly or quarterly) to quarterly frequency.

    The resulting index uses the start of the quarter (``QS``).
    """
    func = _AGG_FUNCS.get(agg)
    if func is None:
        raise ValueError(f"Invalid aggregation: {agg}")
    resampled = series.resample("QS").agg(func)
    return resampled.dropna()


def transform_indicator(series: pd.Series, spec) -> pd.Series:
    """Converts a monthly/quarterly indicator into the model's quarterly feature.

    Follows ``spec.transform``:
      - "growth": % q/q change of the seasonally adjusted level (same scale as the target);
      - "rate":   quarterly sum (series already in % change, e.g. IPCA);
      - "level":  raw quarterly level.
    """
    q = to_quarterly(series, spec.agg)
    if spec.transform == "growth":
        return (q.pct_change(1) * 100).dropna()
    if spec.transform == "rate":
        return q.dropna()  # to_quarterly already summed the months (agg="sum")
    return q.dropna()


def stl_sa_growth(series: pd.Series, kind: str = "qoq") -> pd.Series:
    """Seasonally adjusts a raw monthly level (NSA) via STL and returns the
    quarterly (% q/q) change of the seasonally adjusted level.

    Uses an OWN seasonal adjustment (robust STL over the log, period 12),
    distinct from the BCB's. Combined with the official seasonally adjusted
    series in a bridge, the divergence between the two methods carries signal
    about GDP seasonality (the main source of divergence IBC-Br x GDP; cf. BCB
    Estudo Especial 3/2018).
    """
    from statsmodels.tsa.seasonal import STL

    s = series.dropna().astype(float)
    s.index = pd.to_datetime(s.index)
    lg = np.log(s)
    res = STL(lg, period=12, robust=True).fit()
    sa_level = np.exp(lg - res.seasonal)
    sa_q = to_quarterly(sa_level, "mean")
    n = 1 if kind == "qoq" else 4
    return (sa_q.pct_change(n) * 100).dropna()


def pib_growth(pib_index: pd.Series, kind: str = "qoq") -> pd.Series:
    """Computes GDP growth from the quarterly index.

    Args:
        pib_index: quarterly index (e.g. chained seasonally adjusted).
        kind: "qoq" (q/q-1 change, %) or "yoy" (q/q-4, %).
    """
    if kind == "qoq":
        return (pib_index.pct_change(1) * 100).dropna()
    if kind == "yoy":
        return (pib_index.pct_change(4) * 100).dropna()
    raise ValueError("kind must be 'qoq' or 'yoy'")


def monthly_growth_features(series: pd.Series, prefix: str) -> pd.DataFrame:
    """% m/m changes of the seasonally adjusted monthly level, pivoted into
    columns {prefix}_m1/{prefix}_m2/{prefix}_m3 per quarter (QS index)."""
    g = (series.pct_change(1) * 100).dropna()
    pos = (g.index.month - 1) % 3 + 1
    quarter = g.index.to_period("Q").to_timestamp()
    df = pd.DataFrame({"q": quarter, "pos": pos, "val": g.values})
    wide = df.pivot_table(index="q", columns="pos", values="val", aggfunc="first")
    wide = wide.reindex(columns=[1, 2, 3])
    wide.columns = [f"{prefix}_m{m}" for m in (1, 2, 3)]
    wide.index.name = None
    return wide.sort_index()


COVID_QUARTERS =["2020-01-01", "2020-04-01", "2020-07-01", "2020-10-01"]


def covid_dummies(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Intervention (pulse) dummies for the quarters of the COVID shock.

    Quarterly GDP growth had extreme outliers in 2020 (a drop of ~-9% and a
    rebound of ~+8%). Treating them as exogenous pulses prevents these points
    from contaminating the parameter estimation in the other quarters
    (structural break). Each column is 1 in the corresponding quarter, 0
    otherwise; in future periods (forecast) they are 0.
    """
    idx = pd.DatetimeIndex(index)
    data = {}
    for q in COVID_QUARTERS:
        ts = pd.Timestamp(q)
        col = f"covid_{ts.year}q{ts.quarter}"
        data[col] = (idx == ts).astype(float)
    return pd.DataFrame(data, index=idx)


COVID_PEAK_QUARTERS = ["2020-01-01", "2020-04-01"]  # 2020Q1 and 2020Q2


def covid_peak_dummy(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Single dummy (1 column) marking the peak of the COVID shock: 2020Q1-Q2.

    Unlike ``covid_dummies`` (one pulse per 2020 quarter), here there is a single
    binary variable equal to 1 in 2020Q1 and 2020Q2 and 0 elsewhere — used as an
    exogenous regressor in the VARX[E]. In future periods it is 0.
    """
    idx = pd.DatetimeIndex(index)
    peak = pd.DatetimeIndex([pd.Timestamp(q) for q in COVID_PEAK_QUARTERS])
    return pd.DataFrame({"covid_peak": idx.isin(peak).astype(float)}, index=idx)


def pca_first_factor(df: pd.DataFrame, name: str = "pca") -> pd.Series:
    """First principal component (standardized) of the columns of ``df``.

    Standardizes each column (z-score) over the rows without NaN, extracts the
    1st PC via SVD and fixes the sign to correlate positively with the mean of
    the columns (so a high factor value = high activity). Returns a series
    aligned to the original index (NaN where some input was missing).
    """
    sub = df.dropna()
    if sub.empty:
        return pd.Series(np.nan, index=df.index, name=name)
    mu = sub.mean()
    sd = sub.std(ddof=0).replace(0.0, 1.0)
    z = (sub - mu) / sd
    u, s, vt = np.linalg.svd(z.values, full_matrices=False)
    pc1 = u[:, 0] * s[0]
    # sign: positive when the inputs are above the mean
    if np.corrcoef(pc1, z.values.mean(axis=1))[0, 1] < 0:
        pc1 = -pc1
    factor = pd.Series(pc1, index=sub.index, name=name)
    return factor.reindex(df.index)


def _period_end(quarter_start: pd.Timestamp) -> pd.Timestamp:
    """Last day of the quarter whose start is ``quarter_start``."""
    return quarter_start + pd.offsets.QuarterEnd(0)


def available_as_of(
    series: pd.Series, spec: config.SeriesSpec, as_of: pd.Timestamp
) -> pd.Series:
    """Filters a quarterly series to what would have been published at ``as_of``.

    A value referring to a quarter is only available when
    ``end_of_quarter + publication_lag_days <= as_of``. This avoids the use of
    future information (look-ahead bias).
    """
    if as_of is None:
        return series
    lag = pd.Timedelta(days=spec.publication_lag_days)
    pub_dates = series.index.map(lambda q: _period_end(q) + lag)
    mask = pub_dates <= as_of
    return series[mask]


def build_dataset(
    raw: dict[str, pd.Series],
    target_kind: str = "qoq",
    as_of: pd.Timestamp | None = None,
    respect_publication_lag: bool = False,
) -> pd.DataFrame:
    """Builds the quarterly ``DataFrame`` with target + indicators.

    Args:
        raw: dictionary name->raw series (see ``data_sources.load_all``).
        target_kind: type of GDP growth ("qoq" or "yoy").
        as_of: reference date to simulate availability. If None, uses
            everything (naive / revised scenario).
        respect_publication_lag: if True, applies the publication lags
            ("realistic" scenario). If False, uses only the temporal cutoff by
            ``as_of`` without lag (scenario with look-ahead bias).

    Returns:
        DataFrame indexed by quarter with column "pib_growth" (target) and one
        column per indicator.
    """
    specs = {spec.name: spec for spec in config.ALL_SERIES}

    # Target
    pib_q = to_quarterly(raw["pib"], specs["pib"].agg)
    target = pib_growth(pib_q, kind=target_kind)
    target_spec = specs["pib"]
    if as_of is not None and respect_publication_lag:
        target = available_as_of(target, target_spec, as_of)
    elif as_of is not None:
        target = target[target.index <= as_of]

    cols = {"pib_growth": target}

    for spec in config.INDICATORS:
        q = to_quarterly(raw[spec.name], spec.agg)
        if as_of is not None and respect_publication_lag:
            q = available_as_of(q, spec, as_of)
        elif as_of is not None:
            q = q[q.index <= as_of]
        cols[spec.name] = q

    df = pd.DataFrame(cols).sort_index()
    return df
