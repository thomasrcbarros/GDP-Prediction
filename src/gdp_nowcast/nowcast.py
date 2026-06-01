"""Pipeline ponta-a-ponta de nowcasting do PIB."""
from __future__ import annotations

import pandas as pd

import config

from . import backtest as bt
from . import data_sources, dataset
from .models import MODELS

# Modelos univariados (só usam o passado do PIB); recebem dummies de COVID como
# intervenção. Os demais usam os indicadores antecedentes.
UNIVARIATE = {"arima", "sarima"}


def load_data(refresh: bool = False) -> dict[str, pd.Series]:
    return data_sources.load_all(refresh=refresh)


def current_dataset(
    raw: dict[str, pd.Series],
    target_kind: str = "qoq",
    as_of: pd.Timestamp | None = None,
    respect_publication_lag: bool = False,
) -> pd.DataFrame:
    return dataset.build_dataset(
        raw,
        target_kind=target_kind,
        as_of=as_of,
        respect_publication_lag=respect_publication_lag,
    )


def _prepare(raw: dict[str, pd.Series], target_kind: str):
    """Prepara as entradas trimestrais.

    Retorna:
        target_full: crescimento do PIB com todo o histórico (ex.: 2003+).
        indicators_q: indicadores trimestrais (pode incluir o trimestre
            seguinte ao último PIB — base do nowcasting via bridge).
    """
    indicators_q = pd.DataFrame(
        {spec.name: dataset.to_quarterly(raw[spec.name], spec.agg) for spec in config.INDICATORS}
    ).sort_index()
    pib_q = dataset.to_quarterly(raw["pib"], config.PIB_TARGET.agg)
    target_full = dataset.pib_growth(pib_q, kind=target_kind).dropna().rename("pib_growth")
    return target_full, indicators_q


def _model_exog(name: str, model_index: pd.DatetimeIndex, indicators_q: pd.DataFrame):
    """Monta a matriz exógena apropriada para cada modelo.

    - arima/sarima: dummies de intervenção da COVID;
    - var: indicadores (tratados como endógenos pelo VAR);
    - bridge: indicadores contemporâneos + dummies de COVID.
    """
    covid = dataset.covid_dummies(model_index)
    if name in UNIVARIATE:
        return covid
    if name == "var":
        return indicators_q.reindex(model_index)
    if name == "bridge":
        ind = indicators_q.reindex(model_index)
        return pd.concat([ind, covid], axis=1)
    return None


def nowcast(
    raw: dict[str, pd.Series],
    model_name: str = "arima",
    target_kind: str = "qoq",
) -> dict:
    """Gera o nowcast do próximo trimestre do PIB."""
    target_full, indicators_q = _prepare(raw, target_kind)
    next_q = target_full.index.max() + pd.offsets.QuarterBegin(1, startingMonth=1)

    model = MODELS[model_name]()
    if model_name in UNIVARIATE:
        exog = dataset.covid_dummies(target_full.index)
        model.fit(target_full, exog)
        fut = dataset.covid_dummies(pd.DatetimeIndex([next_q]))
        fc = model.forecast(steps=1, exog_future=fut)
    elif model_name == "var":
        aligned = pd.concat([target_full, indicators_q], axis=1).dropna()
        model.fit(aligned["pib_growth"], aligned[indicators_q.columns])
        fc = model.forecast(steps=1)
    elif model_name == "bridge":
        idx = pd.DatetimeIndex(list(target_full.index) + [next_q])
        exog = _model_exog("bridge", idx, indicators_q)
        train = pd.concat([target_full, exog.loc[target_full.index]], axis=1).dropna()
        model.fit(train["pib_growth"], train[exog.columns])
        # indicadores do trimestre a prever (já publicados antes do PIB)
        fut = exog.loc[[next_q]]
        if fut.isna().any().any():
            raise ValueError(
                f"Indicadores de {next_q.year}Q{next_q.quarter} ainda não disponíveis."
            )
        fc = model.forecast(steps=1, exog_future=fut)
    else:
        raise ValueError(f"Modelo desconhecido: {model_name}")

    return {
        "model": model.summary(),
        "last_observed_quarter": target_full.index.max(),
        "last_observed_value": float(target_full.iloc[-1]),
        "nowcast_quarter": next_q,
        "nowcast_value": float(fc.iloc[0]),
    }


def run_backtests(
    raw: dict[str, pd.Series],
    model_names: list[str],
    target_kind: str = "qoq",
    show_progress: bool = True,
    results_out: dict | None = None,
    exclude_years: list[int] | None = None,
) -> pd.DataFrame:
    """Roda backtest realista vs ingênuo (look-ahead) para os modelos dados.

    Todos os modelos são *avaliados* numa janela de teste comum (definida pela
    disponibilidade dos indicadores), embora os univariados *treinem* com todo o
    histórico do PIB. Inclui baselines (random walk e média) e uma coluna de
    RMSE excluindo os anos de ``exclude_years`` (por padrão 2020, a COVID).

    Se ``results_out`` for fornecido, é preenchido com os ``BacktestResult`` do
    regime realista por modelo (para plotagem).
    """
    if exclude_years is None:
        exclude_years = [2020]

    target_full, indicators_q = _prepare(raw, target_kind)
    aligned = pd.concat([target_full, indicators_q], axis=1).dropna()
    target_var = aligned["pib_growth"]

    # janela de teste comum: 1º trimestre que o VAR/bridge consegue avaliar
    mt = config.MIN_TRAIN_QUARTERS
    if len(target_var) <= mt:
        raise ValueError("Histórico insuficiente para o backtest.")
    score_start = target_var.index[mt]

    def add_row(model, res):
        rows.append(
            {
                "model": model,
                "regime": "realista",
                **res.metrics,
                "rmse_ex_covid": bt.rmse_excluding_years(res, exclude_years),
            }
        )
        if results_out is not None:
            results_out[model] = res

    rows: list[dict] = []
    # baselines (na mesma janela de avaliação)
    add_row("random_walk", bt.random_walk_baseline(target_full, score_start=score_start))
    add_row("media", bt.mean_baseline(target_full, score_start=score_start))

    for name in model_names:
        def factory(n=name):
            return MODELS[n]()

        univ = name in UNIVARIATE
        tgt = target_full if univ else target_var
        exog = _model_exog(name, tgt.index, indicators_q)

        label = f"{name} (realista)" if show_progress else None
        real = bt.rolling_backtest(
            tgt, exog, factory, progress_label=label, score_start=score_start
        )
        naive = bt.naive_lookahead_backtest(tgt, exog, factory)
        rows.append(
            {
                "model": name,
                "regime": "realista",
                **real.metrics,
                "rmse_ex_covid": bt.rmse_excluding_years(real, exclude_years),
            }
        )
        rows.append(
            {
                "model": name,
                "regime": "look-ahead",
                **naive.metrics,
                "rmse_ex_covid": bt.rmse_excluding_years(naive, exclude_years),
            }
        )
        if results_out is not None:
            results_out[name] = real

    return pd.DataFrame(rows)
