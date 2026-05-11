"""
examples/read_sweep.py
~~~~~~~~~~~~~~~~~~~~~~
Simple example for reading the markout sweep output files.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

WILLIAMS_DIR = Path(__file__).parent.parent / "williams"

# ── Load ──────────────────────────────────────────────────────────────────────
full = pd.read_csv(WILLIAMS_DIR / "sweep_full.csv")
by_year = pd.read_csv(WILLIAMS_DIR / "sweep_by_year.csv")

# ── Full period ───────────────────────────────────────────────────────────────
print("── Top 20 by mean_5d (full period) ──")
print(full.nlargest(20, "mean_5d")[["ticker","name","signals","mean_5d","win_rate"]].to_string(index=False))

print("\n── Bottom 10 by mean_5d ──")
print(full.nsmallest(10, "mean_5d")[["ticker","name","signals","mean_5d","win_rate"]].to_string(index=False))

print("\n── Win rate > 60% and at least 10 signals ──")
strong = full[(full["win_rate"] > 0.6) & (full["signals"] >= 10)]
print(strong.sort_values("win_rate", ascending=False).to_string(index=False))

# ── By year ───────────────────────────────────────────────────────────────────
print("\n── Cross-universe average by year ──")
agg = (
    by_year.groupby("year")
    .agg(tickers=("ticker","count"), mean_5d=("mean_5d","mean"), win_rate=("win_rate","mean"))
    .round(4)
)
print(agg.to_string())

print("\n── Best year for a specific ticker ──")
ticker = "600519"
t = by_year[by_year["ticker"] == ticker].sort_values("mean_5d", ascending=False)
print(t.to_string(index=False))

# ── Rank all tickers by mean_5d / std(mean_5d) across years ──────────────────
# Consistency score: high = strong signal that repeats year after year.
# Requires at least 3 years of data so std is meaningful.
print("\n── Consistency ranking: mean_5d / std(mean_5d) across years ──")
stats = (
    by_year.groupby(["ticker", "name"])["mean_5d"]
    .agg(years="count", mean_5d="mean", std_5d="std")
    .reset_index()
)
stats = stats[stats["years"] >= 3].copy()
stats["consistency"] = (stats["mean_5d"] / stats["std_5d"]).round(4)
stats = stats.sort_values("consistency", ascending=False).reset_index(drop=True)
stats.index += 1   # rank from 1
print(stats[["ticker","name","years","mean_5d","std_5d","consistency"]].head(30).to_string())

# ── Monthly breakdown ─────────────────────────────────────────────────────────
print("\n── Cross-universe average by month ──")
by_month = pd.read_csv(WILLIAMS_DIR / "sweep_by_month.csv")
MONTH_NAMES = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
agg_m = (
    by_month.groupby("month")
    .agg(tickers=("ticker","count"), mean_5d=("mean_5d","mean"), win_rate=("win_rate","mean"))
    .round(4)
)
agg_m.index = agg_m.index.map(MONTH_NAMES)
print(agg_m.to_string())

print("\n── Monthly consistency: mean_5d / std(mean_5d) across tickers ──")
month_stats = (
    by_month.groupby(["ticker","name","month"])["mean_5d"]
    .mean().reset_index()
)
# For each ticker: consistency = mean(mean_5d across months) / std(mean_5d across months)
ticker_month = (
    month_stats.groupby(["ticker","name"])["mean_5d"]
    .agg(months="count", mean="mean", std="std")
    .reset_index()
)
ticker_month = ticker_month[ticker_month["months"] >= 3]
ticker_month["month_consistency"] = (ticker_month["mean"] / ticker_month["std"]).round(4)
ticker_month = ticker_month.sort_values("month_consistency", ascending=False).reset_index(drop=True)
ticker_month.index += 1
print(ticker_month[["ticker","name","months","mean","std","month_consistency"]].head(30).to_string())
