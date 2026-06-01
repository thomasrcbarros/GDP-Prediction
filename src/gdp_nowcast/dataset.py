"""Construção do dataset trimestral alinhado para nowcasting.

Responsável por:
- agregar indicadores mensais para frequência trimestral;
- construir a variável-alvo (crescimento do PIB);
- simular a disponibilidade real dos dados numa data de referência
  (defasagens de publicação) -- peça central da avaliação de look-ahead bias.
"""
from __future__ import annotations

import pandas as pd

import config

_AGG_FUNCS = {"mean": "mean", "sum": "sum", "last": "last"}


def to_quarterly(series: pd.Series, agg: str) -> pd.Series:
    """Agrega uma série (mensal ou trimestral) para frequência trimestral.

    O índice resultante usa o início do trimestre (``QS``).
    """
    func = _AGG_FUNCS.get(agg)
    if func is None:
        raise ValueError(f"Agregação inválida: {agg}")
    resampled = series.resample("QS").agg(func)
    return resampled.dropna()


def pib_growth(pib_index: pd.Series, kind: str = "qoq") -> pd.Series:
    """Calcula o crescimento do PIB a partir do índice trimestral.

    Args:
        pib_index: índice (ex.: dessazonalizado encadeado) trimestral.
        kind: "qoq" (variação T/T-1, % ) ou "yoy" (T/T-4, %).
    """
    if kind == "qoq":
        return (pib_index.pct_change(1) * 100).dropna()
    if kind == "yoy":
        return (pib_index.pct_change(4) * 100).dropna()
    raise ValueError("kind deve ser 'qoq' ou 'yoy'")


COVID_QUARTERS = ["2020-01-01", "2020-04-01", "2020-07-01", "2020-10-01"]


def covid_dummies(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Dummies de intervenção (pulso) para os trimestres do choque da COVID.

    O crescimento trimestral do PIB teve outliers extremos em 2020 (queda de
    ~-9% e rebote de ~+8%). Tratá-los como pulsos exógenos evita que esses
    pontos contaminem a estimação dos parâmetros nos demais trimestres
    (quebra estrutural). Cada coluna é 1 no trimestre correspondente, 0 caso
    contrário; nos períodos futuros (previsão) valem 0.
    """
    idx = pd.DatetimeIndex(index)
    data = {}
    for q in COVID_QUARTERS:
        ts = pd.Timestamp(q)
        col = f"covid_{ts.year}q{ts.quarter}"
        data[col] = (idx == ts).astype(float)
    return pd.DataFrame(data, index=idx)


def _period_end(quarter_start: pd.Timestamp) -> pd.Timestamp:
    """Último dia do trimestre cujo início é ``quarter_start``."""
    return quarter_start + pd.offsets.QuarterEnd(0)


def available_as_of(
    series: pd.Series, spec: config.SeriesSpec, as_of: pd.Timestamp
) -> pd.Series:
    """Filtra uma série trimestral para o que estaria publicado em ``as_of``.

    Um valor referente a um trimestre só está disponível quando
    ``fim_do_trimestre + publication_lag_days <= as_of``. Isso evita o uso de
    informação do futuro (look-ahead bias).
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
    """Monta o ``DataFrame`` trimestral com alvo + indicadores.

    Args:
        raw: dicionário nome->série bruta (ver ``data_sources.load_all``).
        target_kind: tipo de crescimento do PIB ("qoq" ou "yoy").
        as_of: data de referência para simular disponibilidade. Se None, usa
            tudo (cenário "ingênuo" / revisado).
        respect_publication_lag: se True, aplica as defasagens de publicação
            (cenário "realista"). Se False, usa apenas o corte temporal por
            ``as_of`` sem defasagem (cenário com look-ahead bias).

    Returns:
        DataFrame indexado por trimestre com coluna "pib_growth" (alvo) e uma
        coluna por indicador.
    """
    specs = {spec.name: spec for spec in config.ALL_SERIES}

    # Alvo
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
