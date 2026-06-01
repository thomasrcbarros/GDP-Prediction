#!/usr/bin/env python3
"""CLI principal: gera nowcast do PIB e roda backtests."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gdp_nowcast import nowcast as nc  # noqa: E402
from gdp_nowcast.models import MODELS  # noqa: E402

ALL = list(MODELS.keys())


def main() -> int:
    parser = argparse.ArgumentParser(description="Nowcasting do PIB brasileiro")
    parser.add_argument(
        "--model", default="all", choices=ALL + ["all"], help="modelo a usar"
    )
    parser.add_argument(
        "--target", default="qoq", choices=["qoq", "yoy"], help="tipo de crescimento"
    )
    parser.add_argument("--backtest", action="store_true", help="roda o backtesting")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="gera gráfico comparando os modelos (implica --backtest)",
    )
    parser.add_argument(
        "--plot-file",
        default="backtest_comparison.png",
        help="caminho do PNG gerado por --plot",
    )
    parser.add_argument(
        "--refresh", action="store_true", help="força nova coleta dos dados"
    )
    args = parser.parse_args()
    run_backtest = args.backtest or args.plot

    models = ALL if args.model == "all" else [args.model]
    raw = nc.load_data(refresh=args.refresh)

    nowcasts: dict[str, float] = {}
    print("\n=== NOWCAST (crescimento do PIB, %s) ===" % args.target)
    for name in models:
        try:
            res = nc.nowcast(raw, model_name=name, target_kind=args.target)
            last_q = res["last_observed_quarter"]
            now_q = res["nowcast_quarter"]
            nowcasts[name] = res["nowcast_value"]
            print(
                f"  {name:7s} [{res['model']}]  "
                f"último obs {last_q.year}Q{last_q.quarter} = "
                f"{res['last_observed_value']:+.2f}%  ->  "
                f"nowcast {now_q.year}Q{now_q.quarter} = {res['nowcast_value']:+.2f}%"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:7s} ERRO: {exc}")

    if run_backtest:
        print("\n=== BACKTEST (realista vs look-ahead) ===")
        print("Progresso (janelas re-treinadas por modelo):")
        results: dict = {}
        table = nc.run_backtests(
            raw, models, target_kind=args.target, results_out=results
        )
        print("\nResultados:")
        with_fmt = table.copy()
        for col in ("rmse", "rmse_ex_covid", "mae", "mape", "bias"):
            if col in with_fmt:
                with_fmt[col] = with_fmt[col].map(lambda v: f"{v:.3f}")
        print(with_fmt.to_string(index=False))
        print(
            "\nNota: o regime 'look-ahead' usa informação futura no ajuste e tende a "
            "subestimar o erro. A diferença para o regime 'realista' é o viés de "
            "look-ahead."
        )

        if args.plot:
            from gdp_nowcast import plotting

            path = plotting.plot_comparison(
                results,
                table,
                args.plot_file,
                nowcasts=nowcasts,
                target_kind=args.target,
            )
            print(f"\nGráfico salvo em: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
