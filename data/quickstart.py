"""
examples/quickstart.py
~~~~~~~~~~~~~~~~~~~~~~
Full working example using real A-share data downloaded via AKShare.

Run:
    cd ashare_bt
    python examples/quickstart.py
"""
import pandas as pd

# 显示所有列
pd.set_option('display.max_columns', None)

# 禁止 DataFrame 按宽度自动换行（尽量在一行内输出）
pd.set_option('display.expand_frame_repr', False)

# 可选：不截断单元格内容
pd.set_option('display.max_colwidth', None)

# 可选：增大输出宽度，避免受终端宽度限制换行
pd.set_option('display.width', 1000)

# 恢复默认（如果需要）
# pd.reset_option('display.max_columns')
# pd.reset_option('display.expand_frame_repr')
# pd.reset_option('display.max_colwidth')
# pd.reset_option('display.width')


import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backtest import Backtest
from data.loader import AKLoader
from builtin import SMACross, RSIStrategy, MACDStrategy
from base import Strategy
from indicators import ema, rsi as _rsi
import numpy as np
import pandas as pd

# ─── 1. Load real data ────────────────────────────────────────────────────────
loader = AKLoader(cache_dir="./data_cache")

print("Loading 贵州茅台 (600519)...")
feed = loader.load("600519", start="2020-01-01", adjust="qfq")
print(feed, "\n")

# ─── 2. Search ────────────────────────────────────────────────────────────────
print("─── Search '银行' ───")
print(loader.search("银行").head(8).to_string(index=False), "\n")

# ─── 3. Single backtest ───────────────────────────────────────────────────────
result = Backtest(feed, SMACross, cash=100_000, fast=5, slow=20).run()
print(result)
print(result.summary().to_string(), "\n")

trades = result.trades_df()
if not trades.empty:
    print("─── Last 5 trades ───")
    print(trades.tail(5)[["entry_date","exit_date","entry_price","exit_price","pnl","return_pct","holding_days"]], "\n")

# ─── 4. Multi-stock ──────────────────────────────────────────────────────────
print("Loading multiple stocks...")
feeds = loader.load_batch(["000001","600036","601318","600519"], start="2020-01-01", adjust="qfq")

rows = []
for sym, f in feeds.items():
    m = Backtest(f, SMACross, cash=100_000, fast=5, slow=20).run().metrics
    rows.append({"Symbol":sym,"Return":f"{m['total_return']:+.2%}","Sharpe":f"{m['sharpe']:.3f}",
                 "Max DD":f"{m['max_drawdown']:.2f}%","Win Rate":f"{m['win_rate']:.1%}","Trades":m["n_trades"]})
print("\n─── SMACross(5,20) comparison ───")
print(pd.DataFrame(rows).to_string(index=False), "\n")

# ─── 5. Index ─────────────────────────────────────────────────────────────────
print("Loading CSI 300...")
csi300 = loader.load_index("000300", start="2020-01-01")
print(csi300, "\n")

# ─── 6. Optimise ─────────────────────────────────────────────────────────────
print("─── Optimising SMA ───")
grid = Backtest(feed, SMACross, cash=100_000).optimise(fast=range(3,15,2), slow=range(10,40,5))
print(grid[["fast","slow","sharpe","total_return","max_drawdown","n_trades"]].head(10).to_string(index=False), "\n")

# ─── 7. Plot ─────────────────────────────────────────────────────────────────
try:
    import matplotlib; matplotlib.use("Agg")
    bh = csi300.data["close"] / csi300.data["close"].iloc[0] * 100_000
    fig = result.plot(benchmark=bh.rename("CSI 300"), show=False)
    fig.savefig("backtest_result.png", dpi=120, facecolor="#0f1520", bbox_inches="tight")
    print("Plot saved → backtest_result.png")
except ImportError:
    print("pip install matplotlib for plotting")
