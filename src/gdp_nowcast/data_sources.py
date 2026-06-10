"""Clients for collecting time series from BCB (SGS) and IBGE (aggregates).

Each public function returns a ``pandas.Series`` indexed by date
(``DatetimeIndex``, start of period), already sorted. There is a CSV caching
layer to ensure offline reproducibility.
"""
from __future__ import annotations

import os
import time

import pandas as pd
import requests

import config


def _request_json(url: str, retries: int = 4, timeout: int = 30, headers: dict | None = None):
    """GET with retry and exponential backoff. Returns decoded JSON."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout, headers=headers)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:  # noqa: PERF203
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_exc}")


def fetch_bcb_series(code: str, start: str | None = None) -> pd.Series:
    """Fetches an SGS/BCB series by its code.

    The API returns ``[{"data": "dd/mm/yyyy", "valor": "x"}, ...]``.
    """
    base = config.BCB_SGS_URL.format(code=code)
    # SGS rejects (406) long requests on daily series: each call covers at most
    # ~10 years. We fetch in windows and concatenate.
    di = pd.to_datetime(start) if start else pd.Timestamp("1990-01-01")
    fim = pd.Timestamp.today().normalize()
    chunks = []
    win_start = di
    while win_start <= fim:
        win_end = min(win_start + pd.DateOffset(years=10) - pd.Timedelta(days=1), fim)
        url = f"{base}&dataInicial={win_start:%d/%m/%Y}&dataFinal={win_end:%d/%m/%Y}"
        # SGS returns 404 for windows before the series start (no data);
        # we treat that as an empty window and move on to the next one.
        try:
            part = _request_json(url)
        except RuntimeError:
            part = None
        if part:
            chunks.extend(part)
        win_start = win_end + pd.Timedelta(days=1)
    data = chunks
    if not data:
        return pd.Series(dtype="float64")
    df = pd.DataFrame(data).drop_duplicates(subset="data")
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    s = df.set_index("data")["valor"].sort_index()
    if start is not None:
        s = s[s.index >= pd.to_datetime(start)]
    s.name = code
    return s


YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


def fetch_yahoo_series(symbol: str, start: str | None = None) -> pd.Series:
    """Fetches the monthly close of a symbol on Yahoo Finance (e.g. ``^BVSP``).

    BCB discontinued the Ibovespa series in SGS, so we use Yahoo as the source.
    Returns a monthly ``pd.Series`` (index at the start of the month) of closes.
    """
    p1 = int(pd.to_datetime(start).timestamp()) if start else 0
    p2 = int(pd.Timestamp.today().timestamp())
    url = f"{YAHOO_CHART_URL.format(symbol=symbol)}?period1={p1}&period2={p2}&interval=1mo"
    data = _request_json(url, headers={"User-Agent": "Mozilla/5.0"})
    result = (data.get("chart", {}).get("result") or [None])[0]
    if not result:
        return pd.Series(dtype="float64")
    ts = result["timestamp"]
    closes = result["indicators"]["quote"][0]["close"]
    idx = pd.to_datetime(ts, unit="s").normalize().to_period("M").to_timestamp()
    s = pd.Series(closes, index=idx).dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s.name = symbol
    return s


def fetch_ibge_aggregate(
    aggregate: str, variable: str, periods: str = "all", classific: str = ""
) -> pd.Series:
    """Fetches a variable from an IBGE aggregate (SIDRA API v3).

    Returns the national series (N1). ``classific`` is the classification filter
    in the API format (e.g. ``11046[56726]|12355[107071]``); empty for none.
    """
    cl = f"&classificacao={classific}" if classific else ""
    url = config.IBGE_AGGREGATE_URL.format(
        aggregate=aggregate, periods=periods, variable=variable, classific=cl
    )
    data = _request_json(url)
    if not data:
        return pd.Series(dtype="float64")
    serie_dict = data[0]["resultados"][0]["series"][0]["serie"]
    records = {}
    for period, value in serie_dict.items():
        try:
            records[_parse_ibge_period(period)] = float(value)
        except (TypeError, ValueError):
            continue  # "..." / "-" => unavailable
    s = pd.Series(records).sort_index()
    s.name = f"ibge_{aggregate}_{variable}"
    return s


def _parse_ibge_period(period: str) -> pd.Timestamp:
    """Converts an IBGE period (YYYYMM monthly or YYYYTT quarterly) to Timestamp."""
    year = int(period[:4])
    suffix = int(period[4:])
    if len(period) == 6 and suffix <= 4:  # quarter 1..4
        month = (suffix - 1) * 3 + 1
        return pd.Timestamp(year=year, month=month, day=1)
    # monthly YYYYMM
    return pd.Timestamp(year=year, month=suffix, day=1)


def _cache_path(name: str) -> str:
    return os.path.join(config.DATA_DIR, f"{name}.csv")


def load_series(spec: config.SeriesSpec, refresh: bool = False) -> pd.Series:
    """Loads a series respecting the local cache.

    If ``refresh`` is False and a CSV exists in ``data/``, reads from disk.
    Otherwise, fetches from the API and writes the cache.
    """
    path = _cache_path(spec.name)
    if not refresh and os.path.exists(path):
        s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        s.name = spec.name
        return s

    if spec.source == "bcb":
        s = fetch_bcb_series(spec.code, start=config.DEFAULT_START)
    elif spec.source == "ibge":
        parts = spec.code.split(":")
        aggregate, variable = parts[0], parts[1]
        classific = parts[2] if len(parts) > 2 else ""
        s = fetch_ibge_aggregate(aggregate, variable, classific=classific)
    elif spec.source == "yahoo":
        s = fetch_yahoo_series(spec.code, start=config.DEFAULT_START)
    else:
        raise ValueError(f"Unknown source: {spec.source}")

    s.name = spec.name
    os.makedirs(config.DATA_DIR, exist_ok=True)
    s.to_frame().to_csv(path)
    return s


def load_all(refresh: bool = False) -> dict[str, pd.Series]:
    """Loads all series defined in ``config.ALL_SERIES``."""
    return {spec.name: load_series(spec, refresh=refresh) for spec in config.ALL_SERIES}
