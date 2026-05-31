"""Configuração central do projeto de nowcasting do PIB.

Define os códigos das séries (BCB SGS / IBGE), suas frequências, o tipo de
agregação mensal->trimestral e as defasagens de publicação usadas para simular
o que estava disponível numa data de referência (núcleo da avaliação de
look-ahead bias).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# Diretórios -----------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")

# Endpoints ------------------------------------------------------------------
BCB_SGS_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados?formato=json"
IBGE_AGGREGATE_URL = (
    "https://servicodados.ibge.gov.br/api/v3/agregados/{aggregate}/periodos/"
    "{periods}/variaveis/{variable}?localidades=N1[all]"
)


@dataclass(frozen=True)
class SeriesSpec:
    """Especificação de uma série temporal.

    Attributes:
        name: identificador curto usado como nome de coluna / arquivo de cache.
        source: "bcb" ou "ibge".
        code: código SGS (para BCB).
        freq: "M" (mensal) ou "Q" (trimestral).
        agg: como agregar mensal->trimestral ("mean", "sum", "last").
        publication_lag_days: dias após o fim do período de referência até a
            publicação do dado (usado para simular disponibilidade real).
        is_target: True para a série-alvo (PIB).
    """

    name: str
    source: str
    code: str
    freq: str
    agg: str = "mean"
    publication_lag_days: int = 30
    is_target: bool = False


# Séries-alvo e indicadores antecedentes -------------------------------------
# Códigos validados contra a API do BCB (SGS).
PIB_TARGET = SeriesSpec(
    name="pib",
    source="bcb",
    code="22109",  # PIB - índice encadeado dessazonalizado (média 1995=100), trimestral
    freq="Q",
    agg="last",
    publication_lag_days=60,  # PIB trimestral sai ~60 dias após o fim do trimestre
    is_target=True,
)

INDICATORS = [
    SeriesSpec(
        name="ibcbr",
        source="bcb",
        code="24364",  # IBC-Br dessazonalizado (proxy mensal do PIB)
        freq="M",
        agg="mean",
        publication_lag_days=45,
    ),
    SeriesSpec(
        name="pim",
        source="bcb",
        code="21859",  # Produção industrial - indústria geral (PIM-PF), índice
        freq="M",
        agg="mean",
        publication_lag_days=35,
    ),
    SeriesSpec(
        name="desocupacao",
        source="bcb",
        code="24369",  # Taxa de desocupação - PNAD Contínua (%)
        freq="M",
        agg="mean",
        publication_lag_days=30,
    ),
]

ALL_SERIES = [PIB_TARGET] + INDICATORS

# Parâmetros de modelagem / backtest -----------------------------------------
SEASONAL_PERIOD = 4  # trimestres por ano
MIN_TRAIN_QUARTERS = 20  # tamanho mínimo da janela inicial de treino no backtest
DEFAULT_START = "2003-01-01"  # início da coleta (PNAD/IBC-Br começam ~2002-2003)
