"""End-to-end GDP nowcasting pipeline.

Supports multiple models and **variable sets** (A/B/C/D, see
``config.FEATURE_SETS``). Univariate models (ARIMA/SARIMA) use only the past of
GDP + COVID dummies; bridge and VAR use the indicators of the chosen set.
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
    """Returns (target_full, feats).

    ``feats`` is a quarterly DataFrame with one column per indicator of
    ``config.SERIES`` already transformed (QoQ/rate/level), which may include the
    quarter following the last GDP (basis of the nowcasting via bridge).
    """
    cols = {}
    for name, spec in config.SERIES.items():
        cols[name] = dataset.transform_indicator(raw[name], spec)
    feats = pd.DataFrame(cols).sort_index()
    pib_q = dataset.to_quarterly(raw["pib"], config.PIB_TARGET.agg)
    target_full = dataset.pib_growth(pib_q, kind=target_kind).dropna().rename("pib_growth")
    feats = pd.concat([feats, dataset.monthly_growth_features(raw["ibcbr"], "ibcbr")], axis=1)
    # own seasonal adjustment (STL) of the raw IBC-Br -> column ibcbr_own (set A2)
    feats["ibcbr_own"] = dataset.stl_sa_growth(raw["ibcbr_nsa"]).reindex(feats.index)
    return target_full, feats


def _variant_label(model_name: str, feature_set: str | None) -> str:
    return model_name if feature_set is None else f"{model_name}[{feature_set}]"


def _uses_covid_dummy(model_name, feature_set) -> bool:
    return (model_name, feature_set) in config.COVID_DUMMY_VARIANTS


def _is_varx_e(model_name, feature_set) -> bool:
    return model_name == "varx" and feature_set == "E"


def _exog_e(index, feats):
    """VARX[E] matrix: endogenous (ibcbr, spread) + true exogenous.

    Exogenous: PCA factor of (pim, pms, pmc), consumer confidence and the COVID
    peak dummy. The extra endogenous variables enter here too — ``VarxModel``
    separates which columns are exogenous via ``exog_cols`` (= ``config.VARX_E_EXOG_COLS``).
    """
    spec = config.VARX_E
    endog = feats[spec["endog"]].reindex(index)
    fator = dataset.pca_first_factor(feats[spec["pca_inputs"]], spec["pca_name"]).reindex(index)
    obs = feats[spec["exog_observed"]].reindex(index)
    covid = dataset.covid_peak_dummy(index)
    return pd.concat([endog, fator, obs, covid], axis=1)


def _exog_for(model_name, feature_set, index, feats):
    """Exogenous matrix of a multivariate model, aligned to ``index``.

    For variants in ``config.COVID_DUMMY_VARIANTS`` it appends the COVID peak
    dummy (2020Q1-Q2) as a true exogenous regressor (column ``covid_peak``).
    """
    if _is_varx_e(model_name, feature_set):
        return _exog_e(index, feats)
    if model_name == "umidas":
        ind = feats[config.UMIDAS_COLS].reindex(index)
        return pd.concat([ind, dataset.covid_dummies(index)], axis=1)
    if model_name == "pool":
        cols = sorted({c for comp in config.POOL_COMPONENTS.values() for c in comp})
        ind = feats[cols].reindex(index)
        return pd.concat([ind, dataset.covid_dummies(index)], axis=1)
    cols = config.FEATURE_SETS[feature_set]
    ind = feats[cols].reindex(index)
    if model_name == "bridge":
        return pd.concat([ind, dataset.covid_dummies(index)], axis=1)
    if _uses_covid_dummy(model_name, feature_set):
        return pd.concat([ind, dataset.covid_peak_dummy(index)], axis=1)
    return ind  # VAR/VARX: indicators are endogenous


def _model_factory(model_name, feature_set):
    """Creates the model factory, injecting exog_cols when there is a COVID dummy."""
    if _is_varx_e(model_name, feature_set):
        return lambda: MODELS[model_name](exog_cols=config.VARX_E_EXOG_COLS)
    if _uses_covid_dummy(model_name, feature_set):
        return lambda: MODELS[model_name](exog_cols=["covid_peak"])
    if model_name == "pool":
        return lambda: MODELS["pool"](components=config.POOL_COMPONENTS)
    return lambda: MODELS[model_name]()


def nowcast(raw, model_name="arima", feature_set=None, target_kind="qoq") -> dict:
    """Generates the GDP nowcast for the next quarter for a (model, set)."""
    target_full, feats = _prepare(raw, target_kind)
    next_q = target_full.index.max() + pd.offsets.QuarterBegin(1, startingMonth=1)

    if model_name in UNIVARIATE:
        model = MODELS[model_name]()
        exog = dataset.covid_dummies(target_full.index)
        model.fit(target_full, exog)
        fc = model.forecast(1, exog_future=dataset.covid_dummies(pd.DatetimeIndex([next_q])))
    elif model_name in ("var", "varx"):
        model = _model_factory(model_name, feature_set)()
        idx = pd.DatetimeIndex(list(target_full.index) + [next_q])
        exog_all = _exog_for(model_name, feature_set, idx, feats)
        aligned = pd.concat([target_full, exog_all.loc[target_full.index]], axis=1).dropna()
        model.fit(aligned["pib_growth"], aligned[exog_all.columns])
        # VARX conditions on the contemporaneous indicators already published for
        # the quarter to forecast; VAR ignores exog_future.
        fut = exog_all.reindex([next_q]) if model_name == "varx" else None
        if model_name == "varx":
            # requires all columns present, except the COVID dummy (0 in the future)
            ind_cols = [c for c in exog_all.columns if c != "covid_peak"]
            if fut[ind_cols].isna().any().any():
                raise ValueError(f"Indicators for {next_q.year}Q{next_q.quarter} unavailable.")
        fc = model.forecast(1, exog_future=fut)
    elif model_name in ("bridge", "umidas", "pool"):
        model = _model_factory(model_name, feature_set)()
        idx = pd.DatetimeIndex(list(target_full.index) + [next_q])
        exog = _exog_for(model_name, feature_set, idx, feats)
        train = pd.concat([target_full, exog.loc[target_full.index]], axis=1).dropna()
        model.fit(train["pib_growth"], train[exog.columns])
        fut = exog.loc[[next_q]]
        if model_name != "pool" and fut.isna().any().any():
            raise ValueError(f"Indicators for {next_q.year}Q{next_q.quarter} unavailable.")
        fc = model.forecast(1, exog_future=fut)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return {
        "model": model.summary(),
        "variant": _variant_label(model_name, feature_set),
        "last_observed_quarter": target_full.index.max(),
        "last_observed_value": float(target_full.iloc[-1]),
        "nowcast_quarter": next_q,
        "nowcast_value": float(fc.iloc[0]),
    }


def _common_score_start(target_full, feats, variants):
    """First evaluable quarter common to all multivariate variants."""
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


COVID_SPLIT = "2020-01-01"  # pre/post-2020 boundary for the RMSE per window


def run_backtests(raw, variants=None, target_kind="qoq", show_progress=True,
                  results_out=None, exclude_years=None, train_window=None) -> pd.DataFrame:
    """Comparative backtest (realistic vs look-ahead) of the model x set combinations.

    ``variants`` is a list of tuples ``(model, set|None)``; by default it
    uses ``config.MODEL_VARIANTS``. All evaluated over a common test window.

    RMSE columns: ``rmse`` (whole window), ``rmse_ex_covid`` (excluding
    ``exclude_years``), ``rmse_pre2020`` and ``rmse_pos2020`` (item 3 of the request).

    ``train_window`` (number of quarters) activates the **rolling estimation
    window** instead of the full sample (item 4). E.g.: ``train_window=20`` ~ 5 years.
    """
    variants = variants or config.MODEL_VARIANTS
    exclude_years = exclude_years if exclude_years is not None else [2020]

    target_full, feats = _prepare(raw, target_kind)
    score_start = _common_score_start(target_full, feats, variants)

    rows: list[dict] = []

    def emit(label, res, regime="realistic"):
        rows.append({
            "model": label, "regime": regime, **res.metrics,
            "rmse_ex_covid": bt.rmse_excluding_years(res, exclude_years),
            "rmse_pre2020": bt.rmse_period(res, end=COVID_SPLIT),
            "rmse_pos2020": bt.rmse_period(res, start=COVID_SPLIT),
        })
        if results_out is not None and regime == "realistic":
            results_out[label] = res

    emit("random_walk", bt.random_walk_baseline(target_full, score_start=score_start))
    emit("mean", bt.mean_baseline(target_full, score_start=score_start))

    for model_name, fs in variants:
        label = _variant_label(model_name, fs)

        if model_name in UNIVARIATE:
            factory = lambda n=model_name: MODELS[n]()  # noqa: E731
            tgt = target_full
            exog = dataset.covid_dummies(target_full.index)
        else:
            factory = _model_factory(model_name, fs)
            cols = config.FEATURE_SETS[fs]
            aligned = pd.concat([target_full, feats[cols]], axis=1).dropna()
            tgt = aligned["pib_growth"]
            exog = _exog_for(model_name, fs, tgt.index, feats)

        plabel = label if show_progress else None
        real = bt.rolling_backtest(tgt, exog, factory, progress_label=plabel,
                                   score_start=score_start, train_window=train_window)
        naive = bt.naive_lookahead_backtest(tgt, exog, factory)
        emit(label, real)
        emit(label, naive, regime="look-ahead")

    return pd.DataFrame(rows)
