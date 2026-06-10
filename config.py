"""Configuração central do projeto de nowcasting do PIB.

Define os códigos das séries (BCB SGS / IBGE), a agregação mensal->trimestral, a
transformação usada como feature, as defasagens de publicação (núcleo da
avaliação de look-ahead bias) e os **conjuntos de variáveis** (A/B/C/D) testados
nos modelos multivariados.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# Diretórios -----------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")

# Endpoints ------------------------------------------------------------------
BCB_SGS_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados?formato=json"
# Para o IBGE o "code" carrega: "agregado:variavel[:classificacao]"
IBGE_AGGREGATE_URL = (
    "https://servicodados.ibge.gov.br/api/v3/agregados/{aggregate}/periodos/"
    "{periods}/variaveis/{variable}?localidades=N1[all]{classific}"
)


@dataclass(frozen=True)
class SeriesSpec:
    """Especificação de uma série temporal.

    Attributes:
        name: identificador curto (nome de coluna / arquivo de cache).
        source: "bcb", "ibge" ou "yahoo".
        code: SGS (bcb), "agregado:variavel[:classificacao]" (ibge) ou símbolo
            do Yahoo Finance (yahoo, ex.: "^BVSP").
        freq: "M" (mensal) ou "Q" (trimestral).
        agg: agregação mensal->trimestral ("mean", "sum", "last").
        transform: como vira feature trimestral:
            "growth" = variação % T/T do nível dessaz (QoQ);
            "rate"   = soma trimestral (p/ séries já em variação %, ex.: IPCA);
            "level"  = nível bruto.
        publication_lag_days: dias após o fim do período até a publicação.
        is_target: True para a série-alvo (PIB).
    """

    name: str
    source: str
    code: str
    freq: str
    agg: str = "mean"
    transform: str = "growth"
    publication_lag_days: int = 30
    is_target: bool = False


# Série-alvo -----------------------------------------------------------------
PIB_TARGET = SeriesSpec(
    name="pib",
    source="bcb",
    code="22109",  # PIB - índice encadeado dessazonalizado (média 1995=100), trimestral
    freq="Q",
    agg="last",
    transform="growth",
    publication_lag_days=60,
    is_target=True,
)

# Registro de indicadores (todos dessazonalizados / consistentes com o alvo) --
SERIES: dict[str, SeriesSpec] = {
    "ibcbr": SeriesSpec("ibcbr", "bcb", "24364", "M", "mean", "growth", 45),
    "pim": SeriesSpec("pim", "bcb", "21859", "M", "mean", "growth", 35),
    # PMS/PMC: número-índice de volume COM ajuste sazonal (IBGE SIDRA)
    "pms": SeriesSpec("pms", "ibge", "8688:7168:11046[56726]|12355[107071]", "M", "mean", "growth", 45),
    "pmc": SeriesSpec("pmc", "ibge", "8880:7170:11046[56734]", "M", "mean", "growth", 45),
    # Bloco macro (variante D do VAR/VARX)
    "ipca": SeriesSpec("ipca", "bcb", "433", "M", "sum", "rate", 10),     # IPCA var% mensal
    "cambio": SeriesSpec("cambio", "bcb", "3698", "M", "mean", "growth", 1),  # R$/US$ venda média
    # Bloco mercado/expectativas (variante E do VARX) — ver VARX_E abaixo.
    #   spread   -> Spread médio das operações de crédito - Total (SGS 20783, p.p.).
    #   confcons -> Índice de Confiança do Consumidor (SGS 4393), mensal.
    "spread": SeriesSpec("spread", "bcb", "20783", "M", "mean", "level", 30),
    "confcons": SeriesSpec("confcons", "bcb", "4393", "M", "mean", "level", 5),
}

INDICATORS = list(SERIES.values())
ALL_SERIES = [PIB_TARGET] + INDICATORS

# Conjuntos de variáveis testados nos modelos multivariados ------------------
FEATURE_SETS: dict[str, list[str]] = {
    "A": ["ibcbr"],                          # IBC-Br (prévia oficial do PIB) -> baseline
    "B": ["pim", "pms", "pmc"],              # setorial (oferta)
    "C": ["ibcbr", "pim", "pms", "pmc"],    # tudo
    "D": ["ibcbr", "ipca", "cambio"],       # bloco macro
    # E: VARX redesenhado (ver VARX_E). Endógenas: pib_growth, ibcbr, spread.
    # Exógenas: fator PCA de (pim,pms,pmc), confiança do consumidor e dummy COVID.
    # Listamos aqui todas as séries-base necessárias (p/ alinhar a janela comum).
    "E": ["ibcbr", "spread", "pim", "pms", "pmc", "confcons"],
}
PRINCIPAL_SET = "A"

# Composição explícita do VARX[E] -------------------------------------------
#   endog        -> indicadores endógenos (além de pib_growth) no sistema VAR;
#                   o PIB é condicionado nesses valores contemporâneos (já
#                   publicados antes do PIB).
#   pca_inputs   -> indicadores de atividade resumidos em 1 fator (PCA) exógeno.
#   pca_name     -> nome da coluna do fator.
#   exog_observed-> exógenas observadas adicionais.
VARX_E = {
    "endog": ["ibcbr", "spread"],
    "pca_inputs": ["pim", "pms", "pmc"],
    "pca_name": "fator_atividade",
    "exog_observed": ["confcons"],
}
# Colunas exógenas verdadeiras do VARX[E] (fator PCA + confiança + dummy COVID).
VARX_E_EXOG_COLS = [VARX_E["pca_name"], *VARX_E["exog_observed"], "covid_peak"]

# U-MIDAS / pool de bridges -------------------------------------------------
# U-MIDAS (Foroni-Marcellino-Schumacher, 2015): meses 1-3 do IBC-Br dentro do
# trimestre como regressores irrestritos da bridge.
UMIDAS_COLS = ["ibcbr_m1", "ibcbr_m2", "ibcbr_m3"]
# Pool com pesos iguais de 3 bridges (forecast combination).
POOL_COMPONENTS = {
    "bridge_ibcbr": ["ibcbr"],
    "umidas_ibcbr": UMIDAS_COLS,
    "bridge_setorial": ["pim", "pms", "pmc"],
}

# Baseline de referência do projeto: o IBC-Br é a prévia oficial do PIB (BCB),
# então a bridge sobre o IBC-Br é o padrão a ser superado pelos demais modelos.
BASELINE_VARIANT = ("bridge", "A")

# Variantes que recebem a dummy de pico da COVID (2020Q1-Q2) como exógena.
COVID_DUMMY_VARIANTS = {("varx", "E")}

# Combinações (modelo, conjunto) avaliadas no backtest comparativo.
# Mantemos a baseline (bridge[A]), ARIMA/SARIMA nos gráficos e SÓ os VARX
# entre os multivariados (VAR puro e bridge B/C foram removidos a pedido).
MODEL_VARIANTS: list[tuple[str, str | None]] = [
    ("arima", None),
    ("sarima", None),
    ("bridge", "A"),   # baseline (IBC-Br)
    ("umidas", "A"),   # U-MIDAS: meses 1-3 do IBC-Br irrestritos
    ("pool", "C"),     # combinação equal-weight de 3 bridges
    ("varx", "B"),
    ("varx", "C"),
    ("varx", "D"),
    ("varx", "E"),     # endóg: pib,ibcbr,spread | exóg: PCA(pim,pms,pmc),confcons,dummy
]

# Parâmetros de modelagem / backtest -----------------------------------------
SEASONAL_PERIOD = 4
MIN_TRAIN_QUARTERS = 20
DEFAULT_START = "2000-01-01"
