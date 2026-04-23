"""
examples/williams_markout_demo.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Demonstrates WilliamsSwings, WilliamsSignals, and Markout
on real A-share data from LocalDataAPI.

Run:
    python examples/williams_markout_demo.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.local_api import LocalDataAPI
from signals.williams_signals import WilliamsSwings, WilliamsSignals, Markout

ROOT = "C:/Users/ttjia/OneDrive/Work/ashare/market_data"   # ← your folder

api = LocalDataAPI(ROOT)

# ── 1. Load price data ────────────────────────────────────────────────────────
df = api.get("600519", start="2015-01-01")     # 贵州茅台, full history
print(f"Loaded {len(df)} bars: {df.index[0].date()} → {df.index[-1].date()}\n")

# ── 2. Identify swing points ──────────────────────────────────────────────────
swings = WilliamsSwings(df).fit()

print("Swing point summary (last 10):")
print(swings.summary().tail(10).to_string(index=False))
print()
print(f"  ST highs: {len(swings.st_highs)}  ST lows: {len(swings.st_lows)}")
print(f"  IT highs: {len(swings.it_highs)}  IT lows: {len(swings.it_lows)}")
print()

# ── 3. Generate entry signals ─────────────────────────────────────────────────
# require_it_trend=True  → signals only when IT trend is confirmed (default)
# require_it_trend=False → every confirmed STH / STL regardless of trend
sigs = WilliamsSignals(swings, require_it_trend=True).fit()

print("Signal summary:")
print(sigs.summary().to_string())
print()
print("Last 10 signals:")
print(sigs.to_dataframe().tail(10).to_string(index=False))
print()

# ── 4. Markout ────────────────────────────────────────────────────────────────
mo = Markout(sigs, horizons=[1, 2, 3, 5, 10, 20]).fit()

print("Overall markout stats (all signals):")
print(mo.stats.to_string())
print()

print("Markout split by BUY vs SHORT:")
print(mo.by_side().to_string())
print()

# ── 5. Plot ───────────────────────────────────────────────────────────────────
mo.plot()

# ── 6. Multi-stock sweep ──────────────────────────────────────────────────────
import pandas as pd

print("\n─── Multi-stock markout sweep (5-day mean return) ───")
tickers = ["600519", "000001", "601318", "600036", "000858"]
rows = []
for ticker in tickers:
    try:
        d = api.get(ticker, start="2015-01-01")
        sw = WilliamsSwings(d).fit()
        sg = WilliamsSignals(sw).fit()
        if not sg.signals:
            continue
        mk = Markout(sg, horizons=[5]).fit()
        m  = mk.stats.loc["ret_5d"]
        rows.append({
            "ticker":   ticker,
            "name":     api.name(ticker),
            "signals":  m["n"],
            "mean_5d":  m["mean_%"],
            "win_rate": m["win_rate"],
            "t_stat":   m["t_stat"],
            "p_value":  m["p_value"],
        })
    except Exception as e:
        print(f"  {ticker} failed: {e}")

print(pd.DataFrame(rows).to_string(index=False))
