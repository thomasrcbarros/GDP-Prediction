#!/usr/bin/env python3
"""Downloads and caches the BCB/IBGE series into ``data/``."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config  # noqa: E402
from gdp_nowcast import data_sources  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Data collection for GDP nowcasting")
    parser.add_argument(
        "--refresh", action="store_true", help="forces a new fetch ignoring the cache"
    )
    args = parser.parse_args()

    for spec in config.ALL_SERIES:
        series = data_sources.load_series(spec, refresh=args.refresh)
        period = (
            f"{series.index.min():%Y-%m} to {series.index.max():%Y-%m}"
            if len(series)
            else "empty"
        )
        print(f"  {spec.name:14s} {len(series):4d} obs  [{period}]  -> data/{spec.name}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
