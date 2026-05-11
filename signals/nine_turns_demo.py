"""
examples/nine_turns_demo.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Demonstrates the Magical Nine Turns (神奇九转) signal with markout analysis.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from data.local_api import LocalDataAPI
from signals.nine_turns import NineTurnsSignals
from signals.williams_signals import Markout

ROOT = "C:/Users/ttjia/OneDrive/Work/ashare/market_data"
api  = LocalDataAPI(ROOT)

# ── 1. Single stock ───────────────────────────────────────────────────────────
df = api.get("600519", start="2015-01-01")
print(f"Loaded {len(df)} bars\n")

# Basic
nt = NineTurnsSignals(df).fit()
print(nt.summary().to_string())
print("\nLast 10 signals:")
print(nt.to_dataframe().tail(10).to_string(index=False))

# With perfection filter
nt_p = NineTurnsSignals(df, perfect=True).fit()
print(f"\nWith perfect filter: {nt_p.summary()['total_signals']} signals "
      f"(vs {nt.summary()['total_signals']} without)")

# ── 2. Markout ────────────────────────────────────────────────────────────────
mo = Markout(nt, horizons=[1, 2, 3, 5, 10, 20]).fit()
print("\n── Markout stats (all signals) ──")
print(mo.stats.to_string())

print("\n── Markout by side ──")
print(mo.by_side().to_string())

# ── 3. Compare perfect vs non-perfect ────────────────────────────────────────
mo_p = Markout(nt_p, horizons=[5]).fit()
mo_b = Markout(nt,   horizons=[5]).fit()

print("\n── Perfect vs Basic (5d mean return) ──")
print(f"  Basic:   mean={mo_b.stats.loc['ret_5d','mean_%']:+.4f}%  "
      f"win_rate={mo_b.stats.loc['ret_5d','win_rate']:.1%}  "
      f"n={int(mo_b.stats.loc['ret_5d','n'])}")
print(f"  Perfect: mean={mo_p.stats.loc['ret_5d','mean_%']:+.4f}%  "
      f"win_rate={mo_p.stats.loc['ret_5d','win_rate']:.1%}  "
      f"n={int(mo_p.stats.loc['ret_5d','n'])}")

# ── 4. Multi-stock sweep ──────────────────────────────────────────────────────
print("\n── Multi-stock Nine Turns sweep (5d, perfect=True) ──")
tickers = ["600519", "000001", "601318", "600036", "000858"]
rows = []
for ticker in tickers:
    try:
        d  = api.get(ticker, start="2015-01-01")
        nt = NineTurnsSignals(d, perfect=True).fit()
        if not nt.signals:
            continue
        mk = Markout(nt, horizons=[5]).fit()
        m  = mk.stats.loc["ret_5d"]
        rows.append({
            "ticker":   ticker,
            "name":     api.name(ticker),
            "signals":  int(m["n"]),
            "mean_5d":  m["mean_%"],
            "win_rate": m["win_rate"],
            "t_stat":   m["t_stat"],
            "p_value":  m["p_value"],
        })
    except Exception as e:
        print(f"  {ticker} failed: {e}")

print(pd.DataFrame(rows).to_string(index=False))

# ── 5. Plot ───────────────────────────────────────────────────────────────────
try:
    mo.plot()
except ImportError:
    print("\npip install matplotlib for plotting")
