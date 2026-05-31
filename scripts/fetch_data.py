#!/usr/bin/env python3
"""Baixa e cacheia as séries do BCB/IBGE em ``data/``."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config  # noqa: E402
from gdp_nowcast import data_sources  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Coleta de dados para nowcasting do PIB")
    parser.add_argument(
        "--refresh", action="store_true", help="força nova busca ignorando o cache"
    )
    args = parser.parse_args()

    for spec in config.ALL_SERIES:
        series = data_sources.load_series(spec, refresh=args.refresh)
        period = (
            f"{series.index.min():%Y-%m} a {series.index.max():%Y-%m}"
            if len(series)
            else "vazio"
        )
        print(f"  {spec.name:14s} {len(series):4d} obs  [{period}]  -> data/{spec.name}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
