"""Time series models for GDP nowcasting."""

from .arima import ArimaModel
from .sarima import SarimaModel
from .var import VarModel
from .varx import VarxModel
from .bridge import BridgeModel
from .pool import PoolModel

MODELS = {
    "arima": ArimaModel,
    "sarima": SarimaModel,
    "var": VarModel,
    "varx": VarxModel,
    "bridge": BridgeModel,
    "umidas": BridgeModel,
    "pool": PoolModel,
}

__all__ = ["ArimaModel", "SarimaModel", "VarModel", "VarxModel", "BridgeModel", "PoolModel", "MODELS"]
