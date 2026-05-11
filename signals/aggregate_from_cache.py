"""
examples/aggregate_from_cache.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Re-aggregates markout results from cached parquet files WITHOUT
re-running swing detection or signal generation.

Requires that markout_sweep.py has been run at least once with
--cache enabled (default), producing williams/signals_cache/*.parquet.

Can produce:
  sweep_full.csv       — full period
  sweep_by_year.csv    — by calendar year
  sweep_by_month.csv   — by calendar month
  sweep_by_quarter.csv — by quarter (Q1-Q4)
  sweep_by_dow.csv     — by day of week (Mon-Fri)

Usage:
    python examples/aggregate_from_cache.py
    python examples/aggregate_from_cache.py --slices year month quarter dow
    python examples/aggregate_from_cache.py --tickers 600519,000001
    python examples/aggregate_from_cache.py --horizon 10
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MONTH_NAMES = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
DOW_NAMES   = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri"}
QUARTER_NAMES = {1:"Q1",2:"Q2",3:"Q3",4:"Q4"}


def agg_slice(group: pd.Series) -> dict:
    s = group.dropna()
    if len(s) < 2:
        return None
    return {
        "n":        len(s),
        "mean_5d":  round(s.mean(), 4),
        "win_rate": round((s > 0).mean(), 4),
    }


def process_cache(cache_dir: Path, horizon: int, tickers: list | None,
                  slices: list[str], outdir: Path) -> None:

    ret_col = f"ret_{horizon}d"
    parquet_files = sorted(cache_dir.glob("*.parquet"))

    if not parquet_files:
        log.error("No parquet files found in %s. Run markout_sweep.py first.", cache_dir)
        return

    if tickers:
        parquet_files = [p for p in parquet_files if p.stem in tickers]

    log.info("Loading %d cached files…", len(parquet_files))

    full_rows    = []
    year_rows    = []
    month_rows   = []
    quarter_rows = []
    dow_rows     = []

    for i, path in enumerate(parquet_files, 1):
        ticker = path.stem
        try:
            raw = pd.read_parquet(path)
        except Exception as e:
            log.warning("Failed to read %s: %s", path, e)
            continue

        if ret_col not in raw.columns:
            # horizon mismatch — skip
            continue

        # Try to get name from sweep_full if available
        name = ticker

        series = raw[ret_col].dropna()
        if len(series) < 2:
            continue

        # ── full ─────────────────────────────────────────────────────────
        if "full" in slices:
            full_rows.append({
                "ticker":   ticker,
                "name":     name,
                "signals":  len(series),
                "mean_5d":  round(series.mean(), 4),
                "win_rate": round((series > 0).mean(), 4),
            })

        # ── year ─────────────────────────────────────────────────────────
        if "year" in slices:
            for year, grp in raw.groupby(raw.index.year)[ret_col]:
                r = agg_slice(grp)
                if r:
                    year_rows.append({"ticker": ticker, "name": name, "year": year, **r})

        # ── month ─────────────────────────────────────────────────────────
        if "month" in slices:
            for month, grp in raw.groupby(raw.index.month)[ret_col]:
                r = agg_slice(grp)
                if r:
                    month_rows.append({"ticker": ticker, "name": name, "month": month, **r})

        # ── quarter ───────────────────────────────────────────────────────
        if "quarter" in slices:
            for quarter, grp in raw.groupby(raw.index.quarter)[ret_col]:
                r = agg_slice(grp)
                if r:
                    quarter_rows.append({"ticker": ticker, "name": name, "quarter": quarter, **r})

        # ── day of week ───────────────────────────────────────────────────
        if "dow" in slices:
            for dow, grp in raw.groupby(raw.index.dayofweek)[ret_col]:
                r = agg_slice(grp)
                if r:
                    dow_rows.append({"ticker": ticker, "name": name, "dow": dow, **r})

        if i % 500 == 0 or i == len(parquet_files):
            log.info("[%d/%d]", i, len(parquet_files))

    # ── attach names from sweep_full.csv if available ─────────────────────────
    sweep_full = outdir / "sweep_full.csv"
    if sweep_full.exists():
        name_map = pd.read_csv(sweep_full, usecols=["ticker","name"])
        name_map["ticker"] = name_map["ticker"].astype(str).str.zfill(6)
        name_map = dict(zip(name_map["ticker"], name_map["name"]))
        for rows in (full_rows, year_rows, month_rows, quarter_rows, dow_rows):
            for r in rows:
                r["name"] = name_map.get(r["ticker"], r["ticker"])

    # ── save & print ──────────────────────────────────────────────────────────
    saves = {
        "full":    (full_rows,    "sweep_full.csv",       ["ticker","name","signals","mean_5d","win_rate"]),
        "year":    (year_rows,    "sweep_by_year.csv",    ["ticker","name","year","n","mean_5d","win_rate"]),
        "month":   (month_rows,   "sweep_by_month.csv",   ["ticker","name","month","n","mean_5d","win_rate"]),
        "quarter": (quarter_rows, "sweep_by_quarter.csv", ["ticker","name","quarter","n","mean_5d","win_rate"]),
        "dow":     (dow_rows,     "sweep_by_dow.csv",     ["ticker","name","dow","n","mean_5d","win_rate"]),
    }

    for key, (rows, filename, cols) in saves.items():
        if key not in slices or not rows:
            continue
        df = pd.DataFrame(rows)
        path = outdir / filename
        df.to_csv(path, index=False)
        log.info("Saved %s  (%d rows)", path, len(df))

    # ── cross-universe summaries ──────────────────────────────────────────────
    if "month" in slices and month_rows:
        df = pd.DataFrame(month_rows)
        agg = (df.groupby("month")
               .agg(tickers=("ticker","count"), mean_5d=("mean_5d","mean"),
                    win_rate=("win_rate","mean"))
               .round(4))
        agg.index = agg.index.map(MONTH_NAMES)
        print("\n── Cross-universe average by Month ──")
        print(agg.to_string())

    if "quarter" in slices and quarter_rows:
        df = pd.DataFrame(quarter_rows)
        agg = (df.groupby("quarter")
               .agg(tickers=("ticker","count"), mean_5d=("mean_5d","mean"),
                    win_rate=("win_rate","mean"))
               .round(4))
        agg.index = agg.index.map(QUARTER_NAMES)
        print("\n── Cross-universe average by Quarter ──")
        print(agg.to_string())

    if "dow" in slices and dow_rows:
        df = pd.DataFrame(dow_rows)
        agg = (df.groupby("dow")
               .agg(tickers=("ticker","count"), mean_5d=("mean_5d","mean"),
                    win_rate=("win_rate","mean"))
               .round(4))
        agg.index = agg.index.map(DOW_NAMES)
        print("\n── Cross-universe average by Day of Week ──")
        print(agg.to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",     default="williams")
    parser.add_argument("--cache",   default="signals_cache",
                        help="Cache subfolder within --out (default: signals_cache)")
    parser.add_argument("--horizon", default=5,    type=int)
    parser.add_argument("--tickers", default=None)
    parser.add_argument("--slices",  default="full year month quarter dow",
                        help="Space-separated list of slices to compute. "
                             "Options: full year month quarter dow")
    args = parser.parse_args()

    if args.out == "williams":
        outdir = Path(__file__).parent.parent / "williams"
    else:
        outdir = Path(args.out)

    cache_dir = outdir / args.cache
    if not cache_dir.exists():
        log.error("Cache folder not found: %s\nRun markout_sweep.py first.", cache_dir)
        return

    tickers = (
        [t.strip().zfill(6) for t in args.tickers.split(",")]
        if args.tickers else None
    )
    slices = args.slices.split()

    log.info("Cache: %s  |  Horizon: %dd  |  Slices: %s", cache_dir, args.horizon, slices)
    process_cache(cache_dir, args.horizon, tickers, slices, outdir)


if __name__ == "__main__":
    main()
