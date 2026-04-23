"""
examples/markout_sweep.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Multi-stock Williams signal markout sweep across the full local universe.

Outputs:
  sweep_full.csv    — one row per ticker, full period since 2015
  sweep_by_year.csv — one row per (ticker x year)

Reports only mean_5d and win_rate.

Usage:
    python examples/markout_sweep.py
    python examples/markout_sweep.py --tickers 600519,000001,601318
    python examples/markout_sweep.py --horizon 10 --workers 4
    python examples/markout_sweep.py --out C:/results
"""

import argparse
import sys
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── worker (top-level for multiprocessing pickling) ──────────────────────────

def _process_ticker(args):
    ticker, root, horizon, start = args
    try:
        from data.local_api import LocalDataAPI
        from signals.williams_signals import WilliamsSwings, WilliamsSignals, Markout

        api = LocalDataAPI(root)
        df  = api.get(ticker, start=start)
        if len(df) < 100:
            return ticker, api.name(ticker), None, []

        # ── full period ───────────────────────────────────────────────────
        def markout_on(slice_df):
            sw = WilliamsSwings(slice_df).fit()
            sg = WilliamsSignals(sw).fit()
            if not sg.signals:
                return None
            mk = Markout(sg, horizons=[horizon], raw=False).fit()
            m  = mk.stats.loc[f"ret_{horizon}d"]
            return {
                "n":        int(m["n"]),
                "mean_5d":  round(float(m["mean_%"]),  4),
                "win_rate": round(float(m["win_rate"]), 4),
            }

        full = markout_on(df)

        # ── by year ───────────────────────────────────────────────────────
        years     = sorted(df.index.year.unique())
        year_rows = []

        for year in years:
            warmup   = f"{year - 1}-07-01"
            df_slice = df[df.index >= pd.Timestamp(warmup)]
            if len(df_slice) < 60:
                continue
            try:
                sw = WilliamsSwings(df_slice).fit()
                sg = WilliamsSignals(sw).fit()
                if not sg.signals:
                    continue
                mk     = Markout(sg, horizons=[horizon], raw=False).fit()
                subset = mk.raw_df.loc[
                    mk.raw_df.index.year == year, f"ret_{horizon}d"
                ].dropna()
                if len(subset) < 2:
                    continue
                year_rows.append({
                    "year":     year,
                    "n":        len(subset),
                    "mean_5d":  round(subset.mean(), 4),
                    "win_rate": round((subset > 0).mean(), 4),
                })
            except Exception:
                continue

        return ticker, api.name(ticker), full, year_rows

    except Exception as e:
        return ticker, "", None, []


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",    default="C:/Users/ttjia/OneDrive/Work/ashare/market_data")
    parser.add_argument("--out",     default="C:/Users/ttjia/OneDrive/Work/ashare/williams",  help="Output folder for CSVs (default: ./williams)")
    parser.add_argument("--horizon", default=5,    type=int)
    parser.add_argument("--start",   default="2015-01-01")
    parser.add_argument("--workers", default=1,    type=int,
                        help="Parallel processes (default 1; try 2-4 on fast machines)")
    parser.add_argument("--tickers", default=None,
                        help="Comma-separated subset, e.g. 600519,000001")
    args = parser.parse_args()

    # Default output is a "williams" subfolder next to this script
    if args.out == "williams":
        outdir = Path(__file__).parent.parent / "williams"
    else:
        outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    from data.local_api import LocalDataAPI
    api = LocalDataAPI(args.root)

    if args.tickers:
        tickers = [t.strip().zfill(6) for t in args.tickers.split(",")]
    else:
        tickers = api.list_tickers()   # only tickers with a local CSV file

    log.info("Universe : %d tickers", len(tickers))
    log.info("Horizon  : %dd  |  start: %s  |  workers: %d",
             args.horizon, args.start, args.workers)

    tasks      = [(t, args.root, args.horizon, args.start) for t in tickers]
    full_rows  = []
    year_rows  = []
    done       = 0
    errors     = 0
    t0         = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_ticker, task): task[0] for task in tasks}

        for future in as_completed(futures):
            ticker_code = futures[future]
            done += 1
            try:
                ticker, name, full, by_year = future.result()
            except Exception as e:
                log.warning("Worker crash %s: %s", ticker_code, e)
                errors += 1
                continue

            if full:
                full_rows.append({
                    "ticker":   ticker,
                    "name":     name,
                    "signals":  full["n"],
                    "mean_5d":  full["mean_5d"],
                    "win_rate": full["win_rate"],
                })

            for yr in by_year:
                year_rows.append({
                    "ticker":   ticker,
                    "name":     name,
                    "year":     yr["year"],
                    "signals":  yr["n"],
                    "mean_5d":  yr["mean_5d"],
                    "win_rate": yr["win_rate"],
                })

            if done % 50 == 0 or done == len(tickers):
                elapsed = time.time() - t0
                rate    = done / elapsed
                eta     = (len(tickers) - done) / rate if rate > 0 else 0
                log.info("[%d/%d]  with_signals=%d  errors=%d  %.0f/min  ETA %.0fm",
                         done, len(tickers), len(full_rows), errors,
                         rate * 60, eta / 60)

    # ── save CSVs ─────────────────────────────────────────────────────────────
    full_df = pd.DataFrame(full_rows).sort_values("mean_5d", ascending=False)
    year_df = pd.DataFrame(year_rows).sort_values(
        ["year", "mean_5d"], ascending=[True, False]
    )

    full_path = outdir / "sweep_full.csv"
    year_path = outdir / "sweep_by_year.csv"
    full_df.to_csv(full_path, index=False)
    year_df.to_csv(year_path, index=False)
    log.info("Saved %s  (%d rows)", full_path, len(full_df))
    log.info("Saved %s  (%d rows)", year_path, len(year_df))

    # ── print summary ─────────────────────────────────────────────────────────
    h = args.horizon
    print("\n" + "=" * 60)
    print(f"FULL PERIOD — top 30 by mean_{h}d")
    print("=" * 60)
    print(full_df.head(30)[["ticker","name","signals","mean_5d","win_rate"]].to_string(index=False))

    print("\n" + "=" * 60)
    print("BY YEAR — cross-universe average (all tickers)")
    print("=" * 60)
    if not year_df.empty:
        agg = (
            year_df.groupby("year")
            .agg(
                tickers  =("ticker",   "count"),
                mean_5d  =("mean_5d",  "mean"),
                win_rate =("win_rate", "mean"),
            )
            .round(4)
        )
        print(agg.to_string())

    print(f"\nProcessed : {done}  |  with signals: {len(full_df)}"
          f"  |  no signals: {done - len(full_df) - errors}"
          f"  |  errors: {errors}"
          f"  |  elapsed: {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
