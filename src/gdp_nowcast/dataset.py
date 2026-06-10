"""Construção do dataset trimestral alinhado para nowcasting.

Responsável por:
- agregar indicadores mensais para frequência trimestral;
- construir a variável-alvo (crescimento do PIB);
- simular a disponibilidade real dos dados numa data de referência
  (defasagens de publicação) -- peça central da avaliação de look-ahead bias.
"""
from __future__ import annotations

import numpy as np
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


def transform_indicator(series: pd.Series, spec) -> pd.Series:
    """Converte um indicador mensal/trimestral na feature trimestral do modelo.

    Segue ``spec.transform``:
      - "growth": variação % T/T do nível dessaz (mesma escala do alvo);
      - "rate":   soma trimestral (séries já em variação %, ex.: IPCA);
      - "level":  nível trimestral bruto.
    """
    q = to_quarterly(series, spec.agg)
    if spec.transform == "growth":
        return (q.pct_change(1) * 100).dropna()
    if spec.transform == "rate":
        return q.dropna()  # to_quarterly já somou os meses (agg="sum")
    return q.dropna()


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


def monthly_growth_features(series: pd.Series, prefix: str) -> pd.DataFrame:
    """Variações % m/m do nível mensal dessaz, pivotadas em colunas
    {prefix}_m1/{prefix}_m2/{prefix}_m3 por trimestre (índice QS)."""
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


COVID_PEAK_QUARTERS = ["2020-01-01", "2020-04-01"]  # 2020Q1 e 2020Q2


def covid_peak_dummy(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Dummy única (1 coluna) marcando o pico do choque da COVID: 2020Q1-Q2.

    Diferente de ``covid_dummies`` (um pulso por trimestre de 2020), aqui há uma
    só variável binária valendo 1 em 2020Q1 e 2020Q2 e 0 no resto — usada como
    regressor exógeno no VARX[E]. Em períodos futuros vale 0.
    """
    idx = pd.DatetimeIndex(index)
    peak = pd.DatetimeIndex([pd.Timestamp(q) for q in COVID_PEAK_QUARTERS])
    return pd.DataFrame({"covid_peak": idx.isin(peak).astype(float)}, index=idx)


def pca_first_factor(df: pd.DataFrame, name: str = "pca") -> pd.Series:
    """Primeiro componente principal (padronizado) das colunas de ``df``.

    Padroniza cada coluna (z-score) sobre as linhas sem NaN, extrai o 1º PC via
    SVD e fixa o sinal para correlacionar positivamente com a média das colunas
    (assim um valor alto do fator = atividade alta). Retorna uma série alinhada
    ao índice original (NaN onde faltava algum insumo).
    """
    sub = df.dropna()
    if sub.empty:
        return pd.Series(np.nan, index=df.index, name=name)
    mu = sub.mean()
    sd = sub.std(ddof=0).replace(0.0, 1.0)
    z = (sub - mu) / sd
    u, s, vt = np.linalg.svd(z.values, full_matrices=False)
    pc1 = u[:, 0] * s[0]
    # sinal: positivo quando os insumos estão acima da média
    if np.corrcoef(pc1, z.values.mean(axis=1))[0, 1] < 0:
        pc1 = -pc1
    factor = pd.Series(pc1, index=sub.index, name=name)
    return factor.reindex(df.index)


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
