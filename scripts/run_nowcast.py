#!/usr/bin/env python3
"""Main CLI: generates the GDP nowcast and runs backtests.

Supports the variable sets A/B/C/D/E (see config.FEATURE_SETS). For a
multivariate model, use --set; with --model all it runs config.MODEL_VARIANTS.
The backtest reports pre/post-2020 RMSE and accepts --train-window (rolling
estimation window) in addition to the full sample.
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
    p = argparse.ArgumentParser(description="Brazilian GDP nowcasting")
    p.add_argument("--model", default="all", choices=ALL + ["all"], help="model")
    p.add_argument("--set", default=config.PRINCIPAL_SET,
                   choices=list(config.FEATURE_SETS), help="variable set (multivariate)")
    p.add_argument("--target", default="qoq", choices=["qoq", "yoy"])
    p.add_argument("--backtest", action="store_true", help="runs the backtesting")
    p.add_argument("--plot", action="store_true", help="generates plot (implies --backtest)")
    p.add_argument("--plot-file", default="backtest_comparison.png")
    p.add_argument("--train-window", type=int, default=None,
                   help="number of quarters in the rolling estimation window "
                        "(e.g.: 20 ≈ 5 years); default = full sample")
    p.add_argument("--refresh", action="store_true", help="forces a new data collection")
    args = p.parse_args()
    run_backtest = args.backtest or args.plot

    variants = _variants(args.model, args.set)
    raw = nc.load_data(refresh=args.refresh)

    nowcasts: dict[str, float] = {}
    print("\n=== NOWCAST (GDP growth, %s) ===" % args.target)
    for model_name, fs in variants:
        label = nc._variant_label(model_name, fs)
        try:
            res = nc.nowcast(raw, model_name=model_name, feature_set=fs, target_kind=args.target)
            nq = res["nowcast_quarter"]
            nowcasts[label] = res["nowcast_value"]
            print(f"  {label:10s} [{res['model']}]  -> "
                  f"{nq.year}Q{nq.quarter} = {res['nowcast_value']:+.2f}%")
        except Exception as exc:  # noqa: BLE001
            print(f"  {label:10s} ERROR: {exc}")

    if run_backtest:
        win = f"rolling window of {args.train_window}Q" if args.train_window else "full sample"
        print(f"\n=== BACKTEST (realistic vs look-ahead) — estimation: {win} ===")
        print("Progress (windows re-trained per variant):")
        results: dict = {}
        table = nc.run_backtests(raw, variants, target_kind=args.target,
                                 results_out=results, train_window=args.train_window)
        print("\nResults:")
        fmt = table.copy()
        for col in ("rmse", "rmse_ex_covid", "rmse_pre2020", "rmse_pos2020", "mae", "mape", "bias"):
            if col in fmt:
                fmt[col] = fmt[col].map(lambda v: f"{v:.3f}")
        cols = ["model", "regime", "rmse", "rmse_pre2020", "rmse_pos2020",
                "rmse_ex_covid", "mae", "bias", "n"]
        cols = [c for c in cols if c in fmt.columns]
        print(fmt[cols].to_string(index=False))
        print("\nNote: rmse_pre2020/pos2020 = error before/after 2020 (item 3). "
              "'look-ahead' uses future information and underestimates the error.")

        if args.plot:
            from gdp_nowcast import plotting
            path = plotting.plot_comparison(results, table, args.plot_file,
                                            nowcasts=nowcasts, target_kind=args.target)
            print(f"\nPlot saved at: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
