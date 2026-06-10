#!/usr/bin/env python3
"""CLI principal: gera nowcast do PIB e roda backtests.

Suporta os conjuntos de variáveis A/B/C/D/E (ver config.FEATURE_SETS). Para um
modelo multivariado, use --set; com --model all roda config.MODEL_VARIANTS.
O backtest reporta RMSE pré/pós-2020 e aceita --train-window (janela rolante de
estimação) além da amostra completa.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config  # noqa: E402
from gdp_nowcast import nowcast as nc  # noqa: E402
from gdp_nowcast.models import MODELS  # noqa: E402

ALL = list(MODELS.keys())


def _variants(model: str, fset: str) -> list[tuple[str, str | None]]:
    if model == "all":
        return config.MODEL_VARIANTS
    if model in nc.UNIVARIATE:
        return [(model, None)]
    return [(model, fset)]


def main() -> int:
    p = argparse.ArgumentParser(description="Nowcasting do PIB brasileiro")
    p.add_argument("--model", default="all", choices=ALL + ["all"], help="modelo")
    p.add_argument("--set", default=config.PRINCIPAL_SET,
                   choices=list(config.FEATURE_SETS), help="conjunto de variáveis (multivariado)")
    p.add_argument("--target", default="qoq", choices=["qoq", "yoy"])
    p.add_argument("--backtest", action="store_true", help="roda o backtesting")
    p.add_argument("--plot", action="store_true", help="gera gráfico (implica --backtest)")
    p.add_argument("--plot-file", default="backtest_comparison.png")
    p.add_argument("--train-window", type=int, default=None,
                   help="nº de trimestres da janela rolante de estimação "
                        "(ex.: 20 ≈ 5 anos); padrão = amostra completa")
    p.add_argument("--refresh", action="store_true", help="força nova coleta dos dados")
    args = p.parse_args()
    run_backtest = args.backtest or args.plot

    variants = _variants(args.model, args.set)
    raw = nc.load_data(refresh=args.refresh)

    nowcasts: dict[str, float] = {}
    print("\n=== NOWCAST (crescimento do PIB, %s) ===" % args.target)
    for model_name, fs in variants:
        label = nc._variant_label(model_name, fs)
        try:
            res = nc.nowcast(raw, model_name=model_name, feature_set=fs, target_kind=args.target)
            nq = res["nowcast_quarter"]
            nowcasts[label] = res["nowcast_value"]
            print(f"  {label:10s} [{res['model']}]  -> "
                  f"{nq.year}Q{nq.quarter} = {res['nowcast_value']:+.2f}%")
        except Exception as exc:  # noqa: BLE001
            print(f"  {label:10s} ERRO: {exc}")

    if run_backtest:
        win = f"janela rolante de {args.train_window}T" if args.train_window else "amostra completa"
        print(f"\n=== BACKTEST (realista vs look-ahead) — estimação: {win} ===")
        print("Progresso (janelas re-treinadas por variante):")
        results: dict = {}
        table = nc.run_backtests(raw, variants, target_kind=args.target,
                                 results_out=results, train_window=args.train_window)
        print("\nResultados:")
        fmt = table.copy()
        for col in ("rmse", "rmse_ex_covid", "rmse_pre2020", "rmse_pos2020", "mae", "mape", "bias"):
            if col in fmt:
                fmt[col] = fmt[col].map(lambda v: f"{v:.3f}")
        cols = ["model", "regime", "rmse", "rmse_pre2020", "rmse_pos2020",
                "rmse_ex_covid", "mae", "bias", "n"]
        cols = [c for c in cols if c in fmt.columns]
        print(fmt[cols].to_string(index=False))
        print("\nNota: rmse_pre2020/pos2020 = erro antes/depois de 2020 (item 3). "
              "'look-ahead' usa informação futura e subestima o erro.")

        if args.plot:
            from gdp_nowcast import plotting
            path = plotting.plot_comparison(results, table, args.plot_file,
                                            nowcasts=nowcasts, target_kind=args.target)
            print(f"\nGráfico salvo em: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
