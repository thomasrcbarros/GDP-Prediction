"""Geração de gráficos comparando os modelos de nowcasting.

Usa o backend ``Agg`` do matplotlib (sem display); salva arquivos PNG.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

_COLORS = {
    "arima": "#1f77b4",
    "sarima": "#ff7f0e",
    "var": "#2ca02c",
    "varx": "#9467bd",
    "bridge": "#d62728",
    "random_walk": "#999999",
    "media": "#bbbbbb",
}


def _color(label: str) -> str:
    """Cor pela base do rótulo (ex.: 'bridge[B]' -> cor de 'bridge')."""
    base = label.split("[")[0]
    return _COLORS.get(base, "#777777")


def plot_comparison(
    results: dict,
    metrics_table: pd.DataFrame,
    out_path: str,
    nowcasts: dict | None = None,
    target_kind: str = "qoq",
) -> str:
    """Gera figura comparando os modelos e salva em ``out_path``.

    Painéis:
      1. Série temporal: PIB observado vs. previsões one-step-ahead (realista)
         de cada modelo no período de backtest.
      2. Barras de RMSE por modelo (regime realista).
      3. Barras de nowcast do próximo trimestre por modelo (se ``nowcasts``).

    Args:
        results: nome do modelo -> BacktestResult (regime realista).
        metrics_table: tabela retornada por ``run_backtests``.
        out_path: caminho do PNG de saída.
        nowcasts: nome do modelo -> valor do nowcast (opcional).
    """
    has_nowcast = bool(nowcasts)
    ncols = 3 if has_nowcast else 2
    fig = plt.figure(figsize=(6 * ncols, 5))

    # --- Painel 1: previsões vs observado --------------------------------
    ax1 = fig.add_subplot(1, ncols, 1)
    actual = None
    for name, res in results.items():
        if actual is None and len(res.actuals):
            actual = res.actuals
    if actual is not None:
        ax1.plot(
            actual.index, actual.values, color="black", lw=2.2, label="observed GDP"
        )
    for name, res in results.items():
        ax1.plot(
            res.predictions.index,
            res.predictions.values,
            lw=1.4,
            alpha=0.9,
            color=_color(name),
            label=name,
        )
    ax1.set_title(f"One-step-ahead backtest ({target_kind} growth)")
    ax1.set_ylabel("GDP growth (%)")
    ax1.axhline(0, color="grey", lw=0.6, ls="--")
    # legenda abaixo do painel (em colunas) para não cobrir a série
    ax1.legend(
        fontsize=7,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        frameon=False,
    )
    ax1.grid(alpha=0.3)

    # --- Painel 2: RMSE por modelo (realista) ----------------------------
    ax2 = fig.add_subplot(1, ncols, 2)
    real = metrics_table[metrics_table["regime"] == "realista"]
    ax2.bar(
        real["model"],
        real["rmse"],
        color=[_color(m) for m in real["model"]],
    )
    ax2.set_title("Backtest RMSE (lower = better)")
    ax2.set_ylabel("RMSE")
    ax2.tick_params(axis="x", rotation=30)
    ax2.grid(alpha=0.3, axis="y")

    # --- Painel 3: nowcast do próximo trimestre --------------------------
    if has_nowcast:
        ax3 = fig.add_subplot(1, ncols, 3)
        names = list(nowcasts.keys())
        vals = [nowcasts[n] for n in names]
        bars = ax3.bar(names, vals, color=[_color(n) for n in names])
        ax3.set_title("Next-quarter nowcast")
        ax3.set_ylabel("Predicted growth (%)")
        ax3.axhline(0, color="grey", lw=0.6, ls="--")
        ax3.tick_params(axis="x", rotation=30)
        for b, v in zip(bars, vals):
            ax3.annotate(
                f"{v:+.2f}%",
                (b.get_x() + b.get_width() / 2, v),
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=9,
            )
        ax3.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    # bbox_inches="tight" garante que a legenda abaixo do painel 1 não seja cortada
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path
