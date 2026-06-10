"""Brazilian GDP nowcasting package.

Core library: BCB SGS / IBGE data clients (``data_sources``), monthly->quarterly
dataset alignment with anti look-ahead handling (``dataset``), forecasting models
(``models``: ARIMA, SARIMA, VAR, VARX, bridge, pool), rolling backtesting
(``backtest``) and the end-to-end pipeline (``nowcast``).
"""

__version__ = "0.1.0"
