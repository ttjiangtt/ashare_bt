"""
download_all.py
~~~~~~~~~~~~~~~
Downloads daily OHLCV data for all A-share stocks and saves each to
{root_folder}/{ticker}.csv

On the first run: downloads full history.
On subsequent runs: appends only the missing rows (typically just today's bar).

Usage:
    python download_all.py --root C:/Users/ttjia/market_data
    python download_all.py --root C:/Users/ttjia/market_data --adjust qfq
    python download_all.py --root C:/Users/ttjia/market_data --start 2015-01-01
    python download_all.py --root C:/Users/ttjia/market_data --workers 4

Arguments:
    --root      Required. Folder where CSVs are saved. Created if it doesn't exist.
    --adjust    Adjustment type: qfq (default), hfq, or empty string for none.
    --start     Start date for initial download (default: 2015-01-01).
    --workers   Parallel download threads (default: 3). Don't go too high or
                you'll get rate-limited by the data source.
    --symbols   Optional comma-separated list of specific tickers to download,
                e.g. --symbols 600519,000001,601318. Defaults to full universe.
    --throttle  Seconds to wait between requests (default: 0.5).
"""

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── ensure ashare_bt is importable ────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))


# ── helpers ───────────────────────────────────────────────────────────────────

def csv_path(root: Path, ticker: str) -> Path:
    return root / f"{ticker}.csv"


def load_existing(path: Path) -> pd.DataFrame:
    """Read existing CSV; return empty DataFrame if file doesn't exist."""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def save_csv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False, date_format="%Y-%m-%d")


def last_trading_date() -> str:
    """Return today's date, or Friday if today is weekend."""
    today = date.today()
    if today.weekday() == 5:   # Saturday
        today -= timedelta(days=1)
    elif today.weekday() == 6: # Sunday
        today -= timedelta(days=2)
    return str(today)


def download_one(
    ticker: str,
    root: Path,
    start: str,
    adjust: str,
    ak,                     # akshare module (passed in to avoid re-importing)
    throttle: float,
) -> tuple[str, str]:
    """
    Download or incrementally update one ticker.
    Returns (ticker, status_message).
    """
    path = csv_path(root, ticker)
    existing = load_existing(path)

    today_str = last_trading_date()

    # Decide fetch window
    if existing.empty:
        fetch_start = start
        mode = "full"
    else:
        last_date = existing["date"].max()
        # Already up to date?
        if pd.Timestamp(last_date).date() >= pd.Timestamp(today_str).date():
            return ticker, "up-to-date"
        fetch_start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        mode = "incremental"

    # Fetch
    try:
        exchange = "sh" if ticker.startswith(("6", "9")) else "sz"
        sina_symbol = f"{exchange}{ticker}"
        adjust_sina = adjust if adjust else None

        # Try East Money first, fall back to Sina
        try:
            df_new = ak.stock_zh_a_hist(
                symbol=ticker,
                period="daily",
                start_date=fetch_start.replace("-", ""),
                end_date=today_str.replace("-", ""),
                adjust=adjust,
            )
            # Rename Chinese columns
            df_new = df_new.rename(columns={
                "日期": "date", "开盘": "open", "最高": "high",
                "最低": "low",  "收盘": "close", "成交量": "volume",
                "成交额": "amount",
            })
        except Exception:
            df_new = ak.stock_zh_a_daily(
                symbol=sina_symbol,
                start_date=fetch_start.replace("-", ""),
                end_date=today_str.replace("-", ""),
                adjust=adjust_sina,
            )

        if df_new is None or df_new.empty:
            return ticker, "no-data"

        df_new["date"] = pd.to_datetime(df_new["date"])

        # Keep only canonical columns
        cols = [c for c in ("date", "open", "high", "low", "close", "volume", "amount")
                if c in df_new.columns]
        df_new = df_new[cols]

        for col in ("open", "high", "low", "close", "volume"):
            if col in df_new.columns:
                df_new[col] = pd.to_numeric(df_new[col], errors="coerce")

        df_new = df_new.sort_values("date").reset_index(drop=True)

        # Merge with existing
        if not existing.empty:
            combined = pd.concat([existing, df_new], ignore_index=True)
            combined = combined.drop_duplicates("date").sort_values("date").reset_index(drop=True)
        else:
            combined = df_new

        save_csv(path, combined)

        n_new = len(df_new)
        return ticker, f"{mode} +{n_new} rows → {len(combined)} total"

    except Exception as e:
        return ticker, f"ERROR: {e}"
    finally:
        time.sleep(throttle)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download all A-share daily data to individual CSVs."
    )
    parser.add_argument("--root",     required=True,       help="Output folder for CSV files")
    parser.add_argument("--adjust",   default="qfq",       help="qfq / hfq / '' (default: qfq)")
    parser.add_argument("--start",    default="2015-01-01", help="Initial download start date")
    parser.add_argument("--workers",  default=1, type=int,  help="Parallel threads (default: 1). Keep at 1 to avoid V8/py_mini_racer crashes.")
    parser.add_argument("--throttle", default=0.5, type=float, help="Seconds between requests")
    parser.add_argument("--symbols",  default=None,
                        help="Comma-separated tickers, e.g. 600519,000001. Default: all.")
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    log.info("Output folder: %s", root.resolve())

    # Import akshare once
    try:
        import akshare as ak
    except ImportError:
        log.error("akshare not installed. Run: pip install akshare")
        sys.exit(1)

    # Get ticker list
    if args.symbols:
        tickers = [s.strip().zfill(6) for s in args.symbols.split(",")]
        log.info("Downloading %d specified tickers", len(tickers))
    else:
        log.info("Loading universe from local name registry…")
        try:
            from data.local_api import LocalDataAPI
            api     = LocalDataAPI(root, name_file=None)
            tickers = api.universe()
            log.info("Universe: %d tickers from local registry", len(tickers))
        except Exception as e:
            log.error("Could not load local universe: %s", e)
            log.error("Tip: run LocalDataAPI once with an internet connection to"
                      " populate _names.csv, or use --symbols to specify tickers.")
            sys.exit(1)

    # Download
    total   = len(tickers)
    done    = 0
    errors  = []
    updated = []
    skipped = []

    # Warm up py_mini_racer / V8 before spawning threads.
    # AKShare uses a JavaScript engine (py_mini_racer) that crashes if
    # multiple threads try to initialise it simultaneously.  One dummy
    # call here forces the singleton to initialise in the main thread.
    if args.workers > 1:
        log.info("Pre-warming AKShare JS engine (required for multi-threading)…")
        try:
            ak.stock_zh_a_hist(
                symbol=tickers[0], period="daily",
                start_date="20240101", end_date="20240102", adjust="qfq",
            )
        except Exception:
            pass  # result doesn't matter — we just need V8 initialised

    log.info("Starting download with %d worker(s)…", args.workers)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                download_one, ticker, root,
                args.start, args.adjust, ak, args.throttle
            ): ticker
            for ticker in tickers
        }
        for future in as_completed(futures):
            ticker, status = future.result()
            done += 1

            if "ERROR" in status:
                errors.append(ticker)
                log.warning("[%d/%d] %-8s  %s", done, total, ticker, status)
            elif "up-to-date" in status:
                skipped.append(ticker)
                log.debug("[%d/%d] %-8s  %s", done, total, ticker, status)
            else:
                updated.append(ticker)
                log.info("[%d/%d] %-8s  %s", done, total, ticker, status)

    # Summary
    log.info("─" * 55)
    log.info("Done.  Updated: %d  |  Skipped: %d  |  Errors: %d",
             len(updated), len(skipped), len(errors))
    if errors:
        log.warning("Failed tickers: %s", ", ".join(errors[:20]))
        # Save error list for re-running
        err_path = root / "_errors.txt"
        err_path.write_text("\n".join(errors))
        log.warning("Error list saved to %s", err_path)


if __name__ == "__main__":
    main()
