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
        source: "bcb" ou "ibge".
        code: SGS (bcb) ou "agregado:variavel[:classificacao]" (ibge).
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
    # Bloco mercado/expectativas (variante E do VARX)
    #   ibovespa -> Ibovespa, fechamento mensal (SGS 7).
    #   confcons -> Índice de Confiança do Consumidor (SGS 4393), mensal.
    #   confserv -> Sondagem de Serviços – Índice de Confiança dessaz (SGS 20339),
    #               mensal, série inicia em 2008-06.
    "ibovespa": SeriesSpec("ibovespa", "bcb", "7", "M", "last", "growth", 1),
    "confcons": SeriesSpec("confcons", "bcb", "4393", "M", "mean", "level", 5),
    "confserv": SeriesSpec("confserv", "bcb", "20339", "M", "mean", "level", 10),
}

INDICATORS = list(SERIES.values())
ALL_SERIES = [PIB_TARGET] + INDICATORS

# Conjuntos de variáveis testados nos modelos multivariados ------------------
FEATURE_SETS: dict[str, list[str]] = {
    "A": ["ibcbr"],                          # IBC-Br (prévia oficial do PIB) -> baseline
    "B": ["pim", "pms", "pmc"],              # setorial (oferta)
    "C": ["ibcbr", "pim", "pms", "pmc"],    # tudo
    "D": ["ibcbr", "ipca", "cambio"],       # bloco macro
    # E: mercado/expectativas; SEM IBC-Br (requisito do usuário).
    "E": ["pim", "pms", "pmc", "ibovespa", "confcons", "confserv"],
}
PRINCIPAL_SET = "A"

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
    ("varx", "B"),
    ("varx", "C"),
    ("varx", "D"),
    ("varx", "E"),     # Ibovespa + confiança (consumidor+serviços) + setorial, dummy COVID
]

# Parâmetros de modelagem / backtest -----------------------------------------
SEASONAL_PERIOD = 4
MIN_TRAIN_QUARTERS = 20
DEFAULT_START = "2000-01-01"
