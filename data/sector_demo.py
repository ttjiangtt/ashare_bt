"""
examples/sector_demo.py
~~~~~~~~~~~~~~~~~~~~~~~~
Downloads and caches all sector classification data.

Run once to populate the cache, then all lookups are instant offline.

Usage:
    python examples/sector_demo.py               # download + show summary
    python examples/sector_demo.py --refresh     # force re-download
    python examples/sector_demo.py --query 600519  # lookup a ticker
    python examples/sector_demo.py --sector 食品饮料  # list sector members
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from data.sector_api import SectorAPI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

CACHE_DIR = Path(__file__).parent.parent / "sector_cache"




def download_all(api: SectorAPI, force: bool = False) -> None:
    """Download and cache all three schemes. Reports what succeeded."""
    print("\n" + "=" * 55)
    print("Downloading sector classifications…")
    print("=" * 55)

    results = {}

    for scheme, label in [("sw", "申万 (SW)"), ("em", "东方财富 (EM)"), ("csrc", "证监会 (CSRC)")]:
        print(f"\n── {label} ──")
        try:
            if scheme == "sw":
                df = api.sw_table(force_refresh=force)
            elif scheme == "em":
                df = api.em_table(force_refresh=force)
            else:
                df = api.csrc_table(force_refresh=force)

            if df.empty:
                print(f"  ✗ No data (connectivity issue — East Money blocked?)")
                results[scheme] = 0
            else:
                print(f"  ✓ {len(df)} tickers cached → {CACHE_DIR}/sector_{scheme}.csv")
                results[scheme] = len(df)
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results[scheme] = 0

    print("\n" + "=" * 55)
    print("Summary:")
    for scheme, n in results.items():
        status = f"{n} tickers" if n > 0 else "FAILED (will show 'Unknown')"
        print(f"  {scheme.upper():<6} {status}")
    print("=" * 55)




def show_summary(api: SectorAPI) -> None:
    """Print a summary of cached data."""
    sw = api.sw_table()
    em = api.em_table()
    cs = api.csrc_table()

    print("\n── Cached data ──")
    print(f"  SW   : {len(sw)} tickers,  {sw['sw_l1'].nunique()} L1 sectors")
    print(f"  EM   : {len(em)} tickers,  {em['em_sector'].nunique() if not em.empty else 0} boards")
    print(f"  CSRC : {len(cs)} tickers,  {cs['csrc_sector'].nunique() if not cs.empty else 0} sectors")

    if not sw.empty:
        print("\n── SW L1 sectors ──")
        counts = sw.groupby("sw_l1")["ticker"].count().sort_values(ascending=False)
        for sector, count in counts.items():
            print(f"  {sector:<14} {count} tickers")




def lookup_ticker(api: SectorAPI, ticker: str) -> None:
    ticker = ticker.strip().zfill(6)
    print(f"\n── Sector lookup: {ticker} ──")
    print(f"  SW  L1 : {api.sector(ticker, scheme='sw',  level=1)}")
    print(f"  SW  L2 : {api.sector(ticker, scheme='sw',  level=2)}")
    print(f"  EM     : {api.sector(ticker, scheme='em')}")
    print(f"  CSRC   : {api.sector(ticker, scheme='csrc')}")




def lookup_sector(api: SectorAPI, sector_name: str) -> None:
    print(f"\n── Tickers in '{sector_name}' ──")
    for level in (1, 2):
        tickers = api.tickers(sector_name, scheme="sw", level=level)
        if tickers:
            print(f"  SW L{level} ({len(tickers)} tickers): {tickers[:20]}")
    em = api.tickers(sector_name, scheme="em")
    if em:
        print(f"  EM    ({len(em)} tickers): {em[:20]}")





def download_per_stock(root: str, cache_dir: Path, throttle: float = 0.3) -> None:
    """
    Build sector_sw.csv by calling stock_individual_info_em() once per ticker.
    Slow (~30 min for 5000 tickers) but works when bulk APIs are blocked.
    Saves progress incrementally so you can resume if interrupted.
    """
    import sys, time
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.local_api import LocalDataAPI

    try:
        import akshare as ak
    except ImportError:
        print("pip install akshare")
        return

    api     = LocalDataAPI(root)
    tickers = api.list_tickers()

    out_path = cache_dir / "sector_sw.csv"
    # Load existing partial results so we can resume
    if out_path.exists():
        done_df  = pd.read_csv(out_path, dtype=str)
        done_set = set(done_df["ticker"].tolist())
        rows     = done_df.to_dict("records")
        print(f"Resuming — {len(done_set)} already done, {len(tickers)-len(done_set)} remaining")
    else:
        done_set, rows = set(), []

    remaining = [t for t in tickers if t not in done_set]
    total     = len(remaining)
    errors    = 0

    print(f"Fetching sector for {total} tickers via stock_individual_info_em…")
    print(f"Estimated time: {total * throttle / 60:.0f}–{total * (throttle+0.5) / 60:.0f} min\n")

    for i, ticker in enumerate(remaining, 1):
        try:
            info = ak.stock_individual_info_em(symbol=ticker)
            # info is a two-column DataFrame with 'item' and 'value'
            info_dict = dict(zip(info["item"], info["value"])) if "item" in info.columns else {}
            sector = str(info_dict.get("行业", ""))
            name   = str(info_dict.get("股票简称", api.name(ticker)))
            rows.append({
                "ticker": ticker,
                "name":   name,
                "sw_l1":  "",     # not available per-stock, left blank
                "sw_l2":  sector, # individual info returns L2-level name
                "sw_l3":  "",
            })
        except Exception as e:
            rows.append({"ticker": ticker, "name": api.name(ticker),
                         "sw_l1": "", "sw_l2": "", "sw_l3": ""})
            errors += 1

        # Save every 100 tickers
        if i % 100 == 0 or i == total:
            pd.DataFrame(rows).to_csv(out_path, index=False)
            rate = i / (i * throttle + 0.01)
            print(f"  [{i}/{total}]  errors={errors}  saved → {out_path.name}")

        time.sleep(throttle)

    # Build sw_l1 from the L2 info using sw_index_second_info if available
    try:
        l2_info = ak.sw_index_second_info()
        # columns: 行业代码, 行业名称, 上级行业
        l2_to_l1 = dict(zip(l2_info["行业名称"], l2_info["上级行业"]))
        df = pd.read_csv(out_path, dtype=str)
        df["sw_l1"] = df["sw_l2"].map(l2_to_l1).fillna("")
        df.to_csv(out_path, index=False)
        print(f"  SW L1 mapped from L2 using sw_index_second_info ✓")
    except Exception as e:
        print(f"  Could not map L1 from L2: {e}")

    print(f"\nDone. {len(rows)} tickers → {out_path}")






def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",       default="C:/Users/ttjia/OneDrive/Work/ashare/market_data")
    parser.add_argument("--refresh",    action="store_true", help="Force re-download")
    parser.add_argument("--query",      default=None,        help="Lookup a ticker, e.g. 600519")
    parser.add_argument("--sector",     default=None,        help="List members, e.g. 食品饮料")
    parser.add_argument("--no-download",action="store_true", help="Skip download, just show cached data")
    parser.add_argument("--per-stock",  action="store_true",
                        help="Build sector cache by calling per-stock API (slow but reliable from UK)")
    parser.add_argument("--throttle",   default=0.3, type=float,
                        help="Seconds between per-stock calls (default 0.3)")
    args = parser.parse_args()

    api = SectorAPI(cache_dir=CACHE_DIR, stale_days=7)

    if getattr(args, "per_stock", False):
        download_per_stock(args.root, CACHE_DIR, args.throttle)
    elif not args.no_download:
        download_all(api, force=args.refresh)

    if args.query:
        lookup_ticker(api, args.query)
    elif args.sector:
        lookup_sector(api, args.sector)
    else:
        show_summary(api)




if __name__ == "__main__":
    main()
