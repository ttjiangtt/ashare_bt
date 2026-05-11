"""
examples/markout_sweep.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Multi-stock markout sweep across the full local universe.
Supports multiple signal types via --signal.

Outputs (in williams/<signal>/ folder):
  sweep_full.csv       — one row per ticker, full period
  sweep_by_year.csv    — one row per (ticker x year)
  sweep_by_month.csv   — one row per (ticker x month)
  signals_cache/       — raw per-signal parquet files

Usage:
    python examples/markout_sweep.py                          # Williams (default)
    python examples/markout_sweep.py --signal nine_turns
    python examples/markout_sweep.py --signal nine_turns --perfect
    python examples/markout_sweep.py --signal williams --tickers 600519,000001
    python examples/markout_sweep.py --signal nine_turns --workers 4
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

MONTH_NAMES = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}


# ── Signal factory (top-level for pickling) ──────────────────────────────────

def _build_signal(signal_name: str, df, signal_kwargs: dict):
    """Instantiate and fit the correct signal class."""
    if signal_name == "williams":
        from signals.williams_signals import WilliamsSwings, WilliamsSignals
        sw = WilliamsSwings(df).fit()
        return WilliamsSignals(sw).fit()
    elif signal_name == "nine_turns":
        from signals.nine_turns import NineTurnsSignals
        return NineTurnsSignals(df, **signal_kwargs).fit()
    else:
        raise ValueError(f"Unknown signal: {signal_name!r}. Choose 'williams' or 'nine_turns'.")


# ── Worker ────────────────────────────────────────────────────────────────────

def _process_ticker(args):
    ticker, root, horizon, start, cache_dir, signal_name, signal_kwargs = args
    try:
        from data.local_api import LocalDataAPI
        from signals.williams_signals import Markout
        from pathlib import Path

        api = LocalDataAPI(root)
        df  = api.get(ticker, start=start)
        if len(df) < 100:
            return ticker, api.name(ticker), None, [], []

        sg = _build_signal(signal_name, df, signal_kwargs)
        if not sg.signals:
            return ticker, api.name(ticker), None, [], []

        mk      = Markout(sg, horizons=[horizon], raw=False).fit()
        raw_df  = mk.raw_df.copy()
        ret_col = f"ret_{horizon}d"

        # ── cache raw returns to parquet ──────────────────────────────────
        if cache_dir:
            try:
                raw_df.to_parquet(Path(cache_dir) / f"{ticker}.parquet")
            except Exception:
                pass

        # ── full period ───────────────────────────────────────────────────
        full_series = raw_df[ret_col].dropna()
        if len(full_series) < 2:
            return ticker, api.name(ticker), None, [], []

        full = {
            "n":        len(full_series),
            "mean_5d":  round(float(full_series.mean()),  4),
            "win_rate": round(float((full_series > 0).mean()), 4),
        }

        # ── by year ───────────────────────────────────────────────────────
        year_rows = []
        for year in sorted(raw_df.index.year.unique()):
            subset = raw_df.loc[raw_df.index.year == year, ret_col].dropna()
            if len(subset) < 2:
                continue
            year_rows.append({
                "year":     year,
                "n":        len(subset),
                "mean_5d":  round(subset.mean(), 4),
                "win_rate": round((subset > 0).mean(), 4),
            })

        # ── by month ──────────────────────────────────────────────────────
        month_rows = []
        for month in range(1, 13):
            subset = raw_df.loc[raw_df.index.month == month, ret_col].dropna()
            if len(subset) < 2:
                continue
            month_rows.append({
                "month":    month,
                "n":        len(subset),
                "mean_5d":  round(subset.mean(), 4),
                "win_rate": round((subset > 0).mean(), 4),
            })

        return ticker, api.name(ticker), full, year_rows, month_rows

    except Exception as e:
        return ticker, "", None, [], []


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",    default="C:/Users/ttjia/OneDrive/Work/ashare/market_data")
    parser.add_argument("--out",     default=None,
                        help="Output base folder. Default: /<signal>/")
    parser.add_argument("--signal",  default="williams",
                        choices=["williams", "nine_turns"],
                        help="Signal to use (default: williams)")
    parser.add_argument("--perfect", action="store_true",
                        help="nine_turns only: apply perfection filter")
    parser.add_argument("--horizon", default=5,    type=int)
    parser.add_argument("--start",   default="2015-01-01")
    parser.add_argument("--workers", default=1,    type=int)
    parser.add_argument("--tickers", default=None)
    parser.add_argument("--cache",   default="signals_cache",
                        help="Cache subfolder name. Set to '' to disable.")
    args = parser.parse_args()

    # ── Output folder: williams/<signal_name>/ ────────────────────────────────
    signal_name = args.signal
    signal_tag  = signal_name if signal_name == "williams" else (
        "nine_turns_perfect" if args.perfect else "nine_turns"
    )
    signal_kwargs = {}
    if signal_name == "nine_turns":
        signal_kwargs["perfect"] = args.perfect

    if args.out:
        outdir = Path(args.out)
    else:
        outdir = Path(__file__).parent.parent / signal_tag
    outdir.mkdir(parents=True, exist_ok=True)

    if args.cache:
        cache_dir = outdir / args.cache
        cache_dir.mkdir(parents=True, exist_ok=True)
        log.info("Signal cache : %s", cache_dir)
    else:
        cache_dir = None

    from data.local_api import LocalDataAPI
    api = LocalDataAPI(args.root)

    tickers = (
        [t.strip().zfill(6) for t in args.tickers.split(",")]
        if args.tickers else api.list_tickers()
    )

    log.info("Signal   : %s  %s", signal_name, "(perfect)" if args.perfect else "")
    log.info("Universe : %d tickers", len(tickers))
    log.info("Horizon  : %dd  |  start: %s  |  workers: %d",
             args.horizon, args.start, args.workers)
    log.info("Output   : %s", outdir)

    tasks      = [(t, args.root, args.horizon, args.start,
                   cache_dir, signal_name, signal_kwargs) for t in tickers]
    full_rows  = []
    year_rows  = []
    month_rows = []
    done = errors = 0
    t0   = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_ticker, task): task[0] for task in tasks}

        for future in as_completed(futures):
            ticker_code = futures[future]
            done += 1
            try:
                ticker, name, full, by_year, by_month = future.result()
            except Exception as e:
                log.warning("Worker crash %s: %s", ticker_code, e)
                errors += 1
                continue

            if full:
                full_rows.append({"ticker": ticker, "name": name,
                                  "signals": full["n"], "mean_5d": full["mean_5d"],
                                  "win_rate": full["win_rate"]})
            for yr in by_year:
                year_rows.append({"ticker": ticker, "name": name,
                                  "year": yr["year"], "n": yr["n"],
                                  "mean_5d": yr["mean_5d"], "win_rate": yr["win_rate"]})
            for mo in by_month:
                month_rows.append({"ticker": ticker, "name": name,
                                   "month": mo["month"], "n": mo["n"],
                                   "mean_5d": mo["mean_5d"], "win_rate": mo["win_rate"]})

            if done % 50 == 0 or done == len(tickers):
                elapsed = time.time() - t0
                rate    = done / elapsed
                eta     = (len(tickers) - done) / rate if rate > 0 else 0
                log.info("[%d/%d]  with_signals=%d  errors=%d  %.0f/min  ETA %.0fm",
                         done, len(tickers), len(full_rows), errors, rate*60, eta/60)

    # ── Save ──────────────────────────────────────────────────────────────────
    full_df  = pd.DataFrame(full_rows).sort_values("mean_5d", ascending=False)
    year_df  = pd.DataFrame(year_rows).sort_values(["year","mean_5d"], ascending=[True,False])
    month_df = pd.DataFrame(month_rows).sort_values(["month","mean_5d"], ascending=[True,False])

    for df, fname in [(full_df,"sweep_full.csv"),(year_df,"sweep_by_year.csv"),(month_df,"sweep_by_month.csv")]:
        p = outdir / fname
        df.to_csv(p, index=False)
        log.info("Saved %s  (%d rows)", p, len(df))

    # ── Print summary ─────────────────────────────────────────────────────────
    h = args.horizon
    print(f"\n{'='*60}\nFULL PERIOD [{signal_tag}] — top 30 by mean_{h}d\n{'='*60}")
    print(full_df.head(30)[["ticker","name","signals","mean_5d","win_rate"]].to_string(index=False))

    print(f"\n{'='*60}\nBY YEAR — cross-universe average\n{'='*60}")
    if not year_df.empty:
        print(year_df.groupby("year")
              .agg(tickers=("ticker","count"),mean_5d=("mean_5d","mean"),win_rate=("win_rate","mean"))
              .round(4).to_string())

    print(f"\n{'='*60}\nBY MONTH — cross-universe average\n{'='*60}")
    if not month_df.empty:
        agg_m = (month_df.groupby("month")
                 .agg(tickers=("ticker","count"),mean_5d=("mean_5d","mean"),win_rate=("win_rate","mean"))
                 .round(4))
        agg_m.index = agg_m.index.map(MONTH_NAMES)
        print(agg_m.to_string())

    print(f"\nProcessed: {done}  |  with signals: {len(full_df)}"
          f"  |  errors: {errors}  |  elapsed: {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
