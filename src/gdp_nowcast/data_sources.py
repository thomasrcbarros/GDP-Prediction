"""Clientes para coleta de séries temporais do BCB (SGS) e IBGE (agregados).

Cada função pública retorna uma ``pandas.Series`` indexada por data
(``DatetimeIndex``, início do período) já ordenada. Há uma camada de cache em
CSV para garantir reprodutibilidade offline.
"""
from __future__ import annotations

import os
import time

import pandas as pd
import requests

import config


def _request_json(url: str, retries: int = 4, timeout: int = 30):
    """GET com retry e backoff exponencial. Retorna JSON decodificado."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:  # noqa: PERF203
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Falha ao buscar {url}: {last_exc}")


def fetch_bcb_series(code: str, start: str | None = None) -> pd.Series:
    """Busca uma série do SGS/BCB pelo código.

    A API retorna ``[{"data": "dd/mm/yyyy", "valor": "x"}, ...]``.
    """
    url = config.BCB_SGS_URL.format(code=code)
    data = _request_json(url)
    if not data:
        return pd.Series(dtype="float64")
    df = pd.DataFrame(data)
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    s = df.set_index("data")["valor"].sort_index()
    if start is not None:
        s = s[s.index >= pd.to_datetime(start)]
    s.name = code
    return s


def fetch_ibge_aggregate(
    aggregate: str, variable: str, periods: str = "all"
) -> pd.Series:
    """Busca uma variável de um agregado do IBGE (API SIDRA v3).

    Retorna a série nacional (N1). Períodos no formato IBGE (ex.: ``202301``
    para trimestre, ``202301`` para mês) são convertidos em datas.
    """
    url = config.IBGE_AGGREGATE_URL.format(
        aggregate=aggregate, periods=periods, variable=variable
    )
    data = _request_json(url)
    if not data:
        return pd.Series(dtype="float64")
    serie_dict = data[0]["resultados"][0]["series"][0]["serie"]
    records = {}
    for period, value in serie_dict.items():
        records[_parse_ibge_period(period)] = pd.to_numeric(value, errors="coerce")
    s = pd.Series(records).sort_index()
    s.name = f"ibge_{aggregate}_{variable}"
    return s


def _parse_ibge_period(period: str) -> pd.Timestamp:
    """Converte período IBGE (YYYYMM mensal ou YYYYTT trimestral) em Timestamp."""
    year = int(period[:4])
    suffix = int(period[4:])
    if len(period) == 6 and suffix <= 4:  # trimestre 1..4
        month = (suffix - 1) * 3 + 1
        return pd.Timestamp(year=year, month=month, day=1)
    # mensal YYYYMM
    return pd.Timestamp(year=year, month=suffix, day=1)


def _cache_path(name: str) -> str:
    return os.path.join(config.DATA_DIR, f"{name}.csv")


def load_series(spec: config.SeriesSpec, refresh: bool = False) -> pd.Series:
    """Carrega uma série respeitando o cache local.

    Se ``refresh`` for False e existir CSV em ``data/``, lê do disco. Caso
    contrário, busca na API e grava o cache.
    """
    path = _cache_path(spec.name)
    if not refresh and os.path.exists(path):
        s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        s.name = spec.name
        return s

    if spec.source == "bcb":
        s = fetch_bcb_series(spec.code, start=config.DEFAULT_START)
    elif spec.source == "ibge":
        aggregate, variable = spec.code.split(":")
        s = fetch_ibge_aggregate(aggregate, variable)
    else:
        raise ValueError(f"Fonte desconhecida: {spec.source}")

    s.name = spec.name
    os.makedirs(config.DATA_DIR, exist_ok=True)
    s.to_frame().to_csv(path)
    return s


def load_all(refresh: bool = False) -> dict[str, pd.Series]:
    """Carrega todas as séries definidas em ``config.ALL_SERIES``."""
    return {spec.name: load_series(spec, refresh=refresh) for spec in config.ALL_SERIES}
