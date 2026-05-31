"""Modelos de séries temporais para nowcasting do PIB."""

from .arima import ArimaModel
from .sarima import SarimaModel
from .var import VarModel

MODELS = {
    "arima": ArimaModel,
    "sarima": SarimaModel,
    "var": VarModel,
}

__all__ = ["ArimaModel", "SarimaModel", "VarModel", "MODELS"]
