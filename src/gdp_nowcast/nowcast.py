"""Pipeline ponta-a-ponta de nowcasting do PIB.

Suporta múltiplos modelos e **conjuntos de variáveis** (A/B/C/D, ver
``config.FEATURE_SETS``). Modelos univariados (ARIMA/SARIMA) usam só o passado do
PIB + dummies de COVID; bridge e VAR usam os indicadores do conjunto escolhido.
"""
from __future__ import annotations

import pandas as pd

import config

from . import backtest as bt
from . import data_sources, dataset
from .models import MODELS

UNIVARIATE = {"arima", "sarima"}


def load_data(refresh: bool = False) -> dict[str, pd.Series]:
    return data_sources.load_all(refresh=refresh)


def current_dataset(raw, target_kind="qoq", as_of=None, respect_publication_lag=False):
    return dataset.build_dataset(
        raw, target_kind=target_kind, as_of=as_of,
        respect_publication_lag=respect_publication_lag,
    )


def _prepare(raw: dict[str, pd.Series], target_kind: str):
    """Retorna (target_full, feats).

    ``feats`` é um DataFrame trimestral com uma coluna por indicador do
    ``config.SERIES`` já transformada (QoQ/rate/level), podendo incluir o
    trimestre seguinte ao último PIB (base do nowcasting via bridge).
    """
    cols = {}
    for name, spec in config.SERIES.items():
        cols[name] = dataset.transform_indicator(raw[name], spec)
    feats = pd.DataFrame(cols).sort_index()
    pib_q = dataset.to_quarterly(raw["pib"], config.PIB_TARGET.agg)
    target_full = dataset.pib_growth(pib_q, kind=target_kind).dropna().rename("pib_growth")
    return target_full, feats


def _variant_label(model_name: str, feature_set: str | None) -> str:
    return model_name if feature_set is None else f"{model_name}[{feature_set}]"


def _exog_for(model_name, feature_set, index, feats):
    """Matriz exógena de um modelo multivariado, alinhada a ``index``."""
    cols = config.FEATURE_SETS[feature_set]
    ind = feats[cols].reindex(index)
    if model_name == "bridge":
        return pd.concat([ind, dataset.covid_dummies(index)], axis=1)
    return ind  # VAR: indicadores são endógenos


def nowcast(raw, model_name="arima", feature_set=None, target_kind="qoq") -> dict:
    """Gera o nowcast do próximo trimestre do PIB para um (modelo, conjunto)."""
    target_full, feats = _prepare(raw, target_kind)
    next_q = target_full.index.max() + pd.offsets.QuarterBegin(1, startingMonth=1)
    model = MODELS[model_name]()

    if model_name in UNIVARIATE:
        exog = dataset.covid_dummies(target_full.index)
        model.fit(target_full, exog)
        fc = model.forecast(1, exog_future=dataset.covid_dummies(pd.DatetimeIndex([next_q])))
    elif model_name in ("var", "varx"):
        cols = config.FEATURE_SETS[feature_set]
        aligned = pd.concat([target_full, feats[cols]], axis=1).dropna()
        model.fit(aligned["pib_growth"], aligned[cols])
        # VARX condiciona nos indicadores contemporâneos já publicados do
        # trimestre a prever; VAR ignora exog_future.
        fut = feats[cols].reindex([next_q]) if model_name == "varx" else None
        if model_name == "varx" and fut.isna().any().any():
            raise ValueError(f"Indicadores de {next_q.year}Q{next_q.quarter} indisponíveis.")
        fc = model.forecast(1, exog_future=fut)
    elif model_name == "bridge":
        idx = pd.DatetimeIndex(list(target_full.index) + [next_q])
        exog = _exog_for("bridge", feature_set, idx, feats)
        train = pd.concat([target_full, exog.loc[target_full.index]], axis=1).dropna()
        model.fit(train["pib_growth"], train[exog.columns])
        fut = exog.loc[[next_q]]
        if fut.isna().any().any():
            raise ValueError(f"Indicadores de {next_q.year}Q{next_q.quarter} indisponíveis.")
        fc = model.forecast(1, exog_future=fut)
    else:
        raise ValueError(f"Modelo desconhecido: {model_name}")

    return {
        "model": model.summary(),
        "variant": _variant_label(model_name, feature_set),
        "last_observed_quarter": target_full.index.max(),
        "last_observed_value": float(target_full.iloc[-1]),
        "nowcast_quarter": next_q,
        "nowcast_value": float(fc.iloc[0]),
    }


def _common_score_start(target_full, feats, variants):
    """Primeiro trimestre avaliável comum a todas as variantes multivariadas."""
    mt = config.MIN_TRAIN_QUARTERS
    starts = [target_full.index[mt]]
    for model_name, fs in variants:
        if fs is None:
            continue
        cols = config.FEATURE_SETS[fs]
        aligned = pd.concat([target_full, feats[cols]], axis=1).dropna()
        if len(aligned) > mt:
            starts.append(aligned.index[mt])
    return max(starts)


def run_backtests(raw, variants=None, target_kind="qoq", show_progress=True,
                  results_out=None, exclude_years=None) -> pd.DataFrame:
    """Backtest comparativo (realista vs look-ahead) das combinações modelo×conjunto.

    ``variants`` é uma lista de tuplas ``(modelo, conjunto|None)``; por padrão
    usa ``config.MODEL_VARIANTS``. Todos avaliados numa janela de teste comum;
    inclui baselines (random walk, média) e coluna ``rmse_ex_covid``.
    """
    variants = variants or config.MODEL_VARIANTS
    exclude_years = exclude_years if exclude_years is not None else [2020]

    target_full, feats = _prepare(raw, target_kind)
    score_start = _common_score_start(target_full, feats, variants)

    rows: list[dict] = []

    def emit(label, res, regime="realista"):
        rows.append({
            "model": label, "regime": regime, **res.metrics,
            "rmse_ex_covid": bt.rmse_excluding_years(res, exclude_years),
        })
        if results_out is not None and regime == "realista":
            results_out[label] = res

    emit("random_walk", bt.random_walk_baseline(target_full, score_start=score_start))
    emit("media", bt.mean_baseline(target_full, score_start=score_start))

    for model_name, fs in variants:
        label = _variant_label(model_name, fs)

        def factory(n=model_name):
            return MODELS[n]()

        if model_name in UNIVARIATE:
            tgt = target_full
            exog = dataset.covid_dummies(target_full.index)
        else:
            cols = config.FEATURE_SETS[fs]
            aligned = pd.concat([target_full, feats[cols]], axis=1).dropna()
            tgt = aligned["pib_growth"]
            exog = _exog_for(model_name, fs, tgt.index, feats)

        plabel = label if show_progress else None
        real = bt.rolling_backtest(tgt, exog, factory, progress_label=plabel,
                                   score_start=score_start)
        naive = bt.naive_lookahead_backtest(tgt, exog, factory)
        emit(label, real)
        emit(label, naive, regime="look-ahead")

    return pd.DataFrame(rows)
