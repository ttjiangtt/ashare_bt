"""
examples/local_api_demo.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Demonstrates all LocalDataAPI features.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.local_api import LocalDataAPI, returns, rolling_vols

ROOT = "C:/Users/ttjia/OneDrive/Work/ashare/market_data"   # ← change this

api = LocalDataAPI(ROOT)
print(api)
print()

# ── Lookup ────────────────────────────────────────────────────────────────────
print(api.name("600519"))             # → 贵州茅台
print(api.ticker("贵州茅台"))         # → 600519
print(api.ticker("茅台"))             # → 600519 (partial match)

print(api.search("银行"))             # DataFrame of banks
print(api.info("600519"))             # metadata dict
print()

# ── Price data ────────────────────────────────────────────────────────────────
df = api.get("600519")                               # full history
df = api.get("600519", start="2023-01-01")           # from date
df = api.get("贵州茅台", start="2023-01-01")         # by name
df = api.get("600519", columns=["close","volume"])   # subset cols

print(df.tail())
print()

# ── Convenience ───────────────────────────────────────────────────────────────
print("Latest close:", api.latest_close("600519"))
print("Latest bar:\n", api.latest("600519"))
print()

# ── Multi-ticker wide table ───────────────────────────────────────────────────
closes = api.get_multi(
    ["600519", "000001", "601318"],
    field="close",
    start="2023-01-01",
)
print(closes.tail())
print()

# ── Universe ──────────────────────────────────────────────────────────────────
all_tickers = api.list_tickers()
sh_tickers  = api.list_tickers(exchange="SH")
print(f"Total: {len(all_tickers)}  |  SH: {len(sh_tickers)}")

all_df = api.list_all()
print(all_df.head(10))
print()

# ── Returns & vol ─────────────────────────────────────────────────────────────
df = api.get("600519", start="2022-01-01")
r  = returns(df)
v  = rolling_vols(df, windows=[5, 20])
print(r.tail())
print(v.tail())
