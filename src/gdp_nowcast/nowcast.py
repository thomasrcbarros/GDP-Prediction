"""Pipeline ponta-a-ponta de nowcasting do PIB."""
from __future__ import annotations

import pandas as pd

import config

from . import backtest as bt
from . import data_sources, dataset
from .models import MODELS


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


def nowcast(
    raw: dict[str, pd.Series],
    model_name: str = "arima",
    target_kind: str = "qoq",
    as_of: pd.Timestamp | None = None,
    respect_publication_lag: bool = True,
) -> dict:
    """Gera o nowcast do próximo trimestre do PIB.

    Respeita por padrão as defasagens de publicação (cenário realista).
    """
    df = current_dataset(
        raw,
        target_kind=target_kind,
        as_of=as_of,
        respect_publication_lag=respect_publication_lag,
    )
    target = df["pib_growth"].dropna()
    exog = df[[s.name for s in config.INDICATORS]]
    exog = exog.loc[: target.index.max()] if not exog.empty else exog

    model = MODELS[model_name]()
    if model_name == "var":
        model.fit(target, exog)
    else:
        model.fit(target)
    fc = model.forecast(steps=1)
    next_q = target.index.max() + pd.offsets.QuarterBegin(1, startingMonth=1)
    return {
        "model": model.summary(),
        "last_observed_quarter": target.index.max(),
        "last_observed_value": float(target.iloc[-1]),
        "nowcast_quarter": next_q,
        "nowcast_value": float(fc.iloc[0]),
    }


def run_backtests(
    raw: dict[str, pd.Series],
    model_names: list[str],
    target_kind: str = "qoq",
    show_progress: bool = True,
) -> pd.DataFrame:
    """Roda backtest realista vs ingênuo (look-ahead) para os modelos dados.

    Com ``show_progress=True`` (padrão) exibe uma barra de progresso por modelo
    no stderr durante o backtest realista (etapa mais demorada).
    """
    df = current_dataset(raw, target_kind=target_kind)
    target = df["pib_growth"].dropna()
    exog = df[[s.name for s in config.INDICATORS]].loc[: target.index.max()]
    # alinhar exog ao target
    aligned = pd.concat([target, exog], axis=1).dropna()
    target = aligned["pib_growth"]
    exog = aligned[[s.name for s in config.INDICATORS]]

    rows = []
    # baseline
    base = bt.random_walk_baseline(target)
    rows.append({"model": "random_walk", "regime": "realista", **base.metrics})

    for name in model_names:
        def factory(n=name):
            return MODELS[n]()

        label = f"{name} (realista)" if show_progress else None
        real = bt.rolling_backtest(
            target, exog if name == "var" else None, factory, progress_label=label
        )
        naive = bt.naive_lookahead_backtest(
            target, exog if name == "var" else None, factory
        )
        rows.append({"model": name, "regime": "realista", **real.metrics})
        rows.append({"model": name, "regime": "look-ahead", **naive.metrics})

    return pd.DataFrame(rows)
