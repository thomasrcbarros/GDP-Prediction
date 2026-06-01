"""Modelos de séries temporais para nowcasting do PIB."""

from .arima import ArimaModel
from .sarima import SarimaModel
from .var import VarModel
from .varx import VarxModel
from .bridge import BridgeModel

MODELS = {
    "arima": ArimaModel,
    "sarima": SarimaModel,
    "var": VarModel,
    "varx": VarxModel,
    "bridge": BridgeModel,
}

__all__ = ["ArimaModel", "SarimaModel", "VarModel", "VarxModel", "BridgeModel", "MODELS"]
