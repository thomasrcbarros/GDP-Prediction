"""Central configuration for the GDP nowcasting project.

Defines the series codes (BCB SGS / IBGE), the monthly->quarterly aggregation, the
transformation used as a feature, the publication lags (core of the
look-ahead bias evaluation), the **variable sets** (A/A2/B/C/D/E) and the
**model variants** evaluated in the comparative backtest.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# Directories ----------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")

# Endpoints ------------------------------------------------------------------
BCB_SGS_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados?formato=json"
# For IBGE the "code" carries: "agregado:variavel[:classificacao]"
IBGE_AGGREGATE_URL = (
    "https://servicodados.ibge.gov.br/api/v3/agregados/{aggregate}/periodos/"
    "{periods}/variaveis/{variable}?localidades=N1[all]{classific}"
)


@dataclass(frozen=True)
class SeriesSpec:
    """Specification of a time series.

    Attributes:
        name: short identifier (column name / cache file).
        source: "bcb", "ibge" or "yahoo".
        code: SGS (bcb), "agregado:variavel[:classificacao]" (ibge) or
            Yahoo Finance symbol (yahoo, e.g.: "^BVSP").
        freq: "M" (monthly) or "Q" (quarterly).
        agg: monthly->quarterly aggregation ("mean", "sum", "last").
        transform: how it becomes a quarterly feature:
            "growth" = % Q/Q change of the seasonally adjusted level (QoQ);
            "rate"   = quarterly sum (for series already in % change, e.g.: IPCA);
            "level"  = raw level.
        publication_lag_days: days after the end of the period until publication.
        is_target: True for the target series (GDP).
    """

    name: str
    source: str
    code: str
    freq: str
    agg: str = "mean"
    transform: str = "growth"
    publication_lag_days: int = 30
    is_target: bool = False


# Target series --------------------------------------------------------------
PIB_TARGET = SeriesSpec(
    name="pib",
    source="bcb",
    code="22109",  # GDP - seasonally adjusted chained index (average 1995=100), quarterly
    freq="Q",
    agg="last",
    transform="growth",
    publication_lag_days=60,
    is_target=True,
)

# Indicator registry (all seasonally adjusted / consistent with the target) --
SERIES: dict[str, SeriesSpec] = {
    "ibcbr": SeriesSpec("ibcbr", "bcb", "24364", "M", "mean", "growth", 45),
    # IBC-Br raw (NSA) — basis for the own seasonal adjustment (STL) of set A2.
    "ibcbr_nsa": SeriesSpec("ibcbr_nsa", "bcb", "24363", "M", "mean", "level", 45),
    "pim": SeriesSpec("pim", "bcb", "21859", "M", "mean", "growth", 35),
    # PMS/PMC: volume index number WITH seasonal adjustment (IBGE SIDRA)
    "pms": SeriesSpec("pms", "ibge", "8688:7168:11046[56726]|12355[107071]", "M", "mean", "growth", 45),
    "pmc": SeriesSpec("pmc", "ibge", "8880:7170:11046[56734]", "M", "mean", "growth", 45),
    # Macro block (variant D of VAR/VARX)
    "ipca": SeriesSpec("ipca", "bcb", "433", "M", "sum", "rate", 10),     # IPCA monthly %change
    "cambio": SeriesSpec("cambio", "bcb", "3698", "M", "mean", "growth", 1),  # R$/US$ average selling rate
    # Market/expectations block (variant E of VARX) — see VARX_E below.
    #   spread   -> Average spread of credit operations - Total (SGS 20783, p.p.).
    #   confcons -> Consumer Confidence Index (SGS 4393), monthly.
    "spread": SeriesSpec("spread", "bcb", "20783", "M", "mean", "level", 30),
    "confcons": SeriesSpec("confcons", "bcb", "4393", "M", "mean", "level", 5),
}

INDICATORS = list(SERIES.values())
ALL_SERIES = [PIB_TARGET] + INDICATORS

# Variable sets tested in the multivariate models ----------------------------
FEATURE_SETS: dict[str, list[str]] = {
    "A": ["ibcbr"],                          # IBC-Br (official GDP preview) -> baseline
    "A2": ["ibcbr", "ibcbr_own"],            # IBC-Br BCB seas. adj. + own seas. adj. (STL)
    "B": ["pim", "pms", "pmc"],              # sectoral (supply)
    "C": ["ibcbr", "pim", "pms", "pmc"],    # everything
    "D": ["ibcbr", "ipca", "cambio"],       # macro block
    # E: redesigned VARX (see VARX_E). Endogenous: pib_growth, ibcbr, spread.
    # Exogenous: PCA factor of (pim,pms,pmc), consumer confidence and COVID dummy.
    # Here we list all the base series needed (to align the common window).
    "E": ["ibcbr", "spread", "pim", "pms", "pmc", "confcons"],
}
PRINCIPAL_SET = "A"

# Explicit composition of VARX[E] -------------------------------------------
#   endog        -> endogenous indicators (besides pib_growth) in the VAR system;
#                   GDP is conditioned on these contemporaneous values (already
#                   published before GDP).
#   pca_inputs   -> activity indicators summarized into 1 exogenous (PCA) factor.
#   pca_name     -> name of the factor column.
#   exog_observed-> additional observed exogenous variables.
VARX_E = {
    "endog": ["ibcbr", "spread"],
    "pca_inputs": ["pim", "pms", "pmc"],
    "pca_name": "fator_atividade",
    "exog_observed": ["confcons"],
}
# True exogenous columns of VARX[E] (PCA factor + confidence + COVID dummy).
VARX_E_EXOG_COLS = [VARX_E["pca_name"], *VARX_E["exog_observed"], "covid_peak"]

# U-MIDAS / pool of bridges -------------------------------------------------
# U-MIDAS (Foroni-Marcellino-Schumacher, 2015): months 1-3 of the IBC-Br within
# the quarter as unrestricted regressors of the bridge.
UMIDAS_COLS = ["ibcbr_m1", "ibcbr_m2", "ibcbr_m3"]
# Equal-weight pool of 3 bridges (forecast combination).
POOL_COMPONENTS = {
    "bridge_ibcbr": ["ibcbr"],
    "umidas_ibcbr": UMIDAS_COLS,
    "bridge_setorial": ["pim", "pms", "pmc"],
}

# Project reference baseline: the IBC-Br is the official GDP preview (BCB),
# so the bridge over the IBC-Br is the standard to be beaten by the other models.
BASELINE_VARIANT = ("bridge", "A")

# Variants that receive the COVID peak dummy (2020Q1-Q2) as exogenous.
COVID_DUMMY_VARIANTS = {("varx", "E")}

# (model, set) combinations evaluated in the comparative backtest.
# We keep the baseline (bridge[A]), ARIMA/SARIMA in the plots and ONLY the VARX
# among the multivariate ones (pure VAR and bridge B/C were removed on request).
MODEL_VARIANTS: list[tuple[str, str | None]] = [
    ("arima", None),
    ("sarima", None),
    ("bridge", "A"),   # baseline (IBC-Br)
    ("bridge", "A2"),  # IBC-Br seas. adj. BCB + own seas. adj. (STL) -> beats the baseline
    ("umidas", "A"),   # U-MIDAS: months 1-3 of the IBC-Br unrestricted
    ("pool", "C"),     # equal-weight combination of 3 bridges
    ("varx", "B"),
    ("varx", "C"),
    ("varx", "D"),
    ("varx", "E"),     # endog: pib,ibcbr,spread | exog: PCA(pim,pms,pmc),confcons,dummy
]

# Modeling / backtest parameters ---------------------------------------------
SEASONAL_PERIOD = 4
MIN_TRAIN_QUARTERS = 20
DEFAULT_START = "2000-01-01"
