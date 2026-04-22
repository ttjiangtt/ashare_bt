"""
examples/quickstart.py
~~~~~~~~~~~~~~~~~~~~~~
Minimal working example showing all major features of ashare_bt.

Run:
    cd ashare_bt
    python examples/quickstart.py
"""

import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ashare_bt import Backtest, DataFeed
from ashare_bt.strategy import SMACross, RSIStrategy, MACDStrategy
from ashare_bt.strategy.base import Strategy
from ashare_bt.utils.indicators import ema, rsi as _rsi


# ─── 1. Generate synthetic A-share data ──────────────────────────────────────────

def synthetic_data(n=1000, seed=42, drift=0.0003, vol=0.015) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    log_ret = rng.normal(drift, vol, n)
    close = 20.0 * np.exp(np.cumsum(log_ret))
    open_ = close * np.exp(rng.normal(0, 0.003, n))
    high  = np.maximum(close, open_) * (1 + rng.uniform(0, 0.008, n))
    low   = np.minimum(close, open_) * (1 - rng.uniform(0, 0.008, n))
    volume = rng.integers(500_000, 5_000_000, n).astype(float)
    return pd.DataFrame({"date": dates, "open": open_, "high": high,
                          "low": low, "close": close, "volume": volume})


df = synthetic_data(n=1000)
feed = DataFeed(df, symbol="000001.SZ")
print(feed)
print()


# ─── 2. Single backtest ───────────────────────────────────────────────────────────

bt = Backtest(
    data=feed,
    strategy=SMACross,
    cash=100_000,
    commission=0.0003,
    stamp_duty=0.001,
    slippage=0.0005,
    fast=5,
    slow=20,
)

result = bt.run()
print(result)
print()
print(result.summary().to_string())
print()

trades = result.trades_df()
if not trades.empty:
    print("─── Last 5 trades ───")
    print(trades.tail(5)[["entry_date", "exit_date", "entry_price",
                            "exit_price", "pnl", "return_pct", "holding_days"]])
    print()


# ─── 3. Multiple strategies comparison ───────────────────────────────────────────

strategies = {
    "SMA(5,20)":     (SMACross,    {"fast": 5, "slow": 20}),
    "SMA(10,30)":    (SMACross,    {"fast": 10, "slow": 30}),
    "RSI(14)":       (RSIStrategy, {}),
    "MACD(12,26,9)": (MACDStrategy, {}),
}

print("─── Strategy comparison ───")
rows = []
for name, (cls, params) in strategies.items():
    r = Backtest(feed, cls, cash=100_000, **params).run()
    m = r.metrics
    rows.append({
        "Strategy":    name,
        "Return":      f"{m['total_return']:+.2%}",
        "Ann. Return": f"{m['ann_return']:+.2%}",
        "Sharpe":      f"{m['sharpe']:.3f}",
        "Max DD":      f"{m['max_drawdown']:.2f}%",
        "Win Rate":    f"{m['win_rate']:.1%}",
        "Trades":      m["n_trades"],
    })
print(pd.DataFrame(rows).to_string(index=False))
print()


# ─── 4. Parameter optimisation ───────────────────────────────────────────────────

print("─── Optimising SMA periods (top 10 by Sharpe) ───")
bt_opt = Backtest(feed, SMACross, cash=100_000)
grid = bt_opt.optimise(fast=range(3, 15, 2), slow=range(10, 40, 5))
cols = ["fast", "slow", "sharpe", "total_return", "max_drawdown", "n_trades"]
print(grid[cols].head(10).to_string(index=False))
print()


# ─── 5. Custom strategy ───────────────────────────────────────────────────────────

class EMAMomentum(Strategy):
    """
    Enter when short EMA > long EMA and RSI just rose above 50.
    Exit when short EMA < long EMA or RSI drops below 50.
    """
    fast: int = 8
    slow: int = 21
    rsi_period: int = 14

    def init(self):
        c = self.data.closes()
        self.fast_ema = self.indicator(ema, c, self.fast)
        self.slow_ema = self.indicator(ema, c, self.slow)
        self.rsi_arr  = self.indicator(_rsi, c, self.rsi_period)

    def next(self):
        i = self.data._cursor
        f = self.fast_ema[i]
        s = self.slow_ema[i]
        r = self.rsi_arr[i]
        if np.isnan(f) or np.isnan(s) or np.isnan(r):
            return

        if self.position is None:
            if f > s and r > 50:
                self.buy(comment="ema_mom_entry")
        else:
            if f < s or r < 50:
                self.sell(comment="ema_mom_exit")


custom_result = Backtest(feed, EMAMomentum, cash=100_000, fast=8, slow=21).run()
print("─── Custom EMAMomentum strategy ───")
print(custom_result.summary()[["总收益率", "Sharpe 比率", "最大回撤", "胜率", "交易次数"]].to_string())
print()

# ─── 6. Plot (requires matplotlib) ───────────────────────────────────────────────

try:
    import matplotlib
    matplotlib.use("Agg")   # headless for CI; remove in interactive sessions
    fig = result.plot(show=False)
    fig.savefig("backtest_result.png", dpi=120, facecolor="#0f1520")
    print("Plot saved to backtest_result.png")
except ImportError:
    print("Install matplotlib to enable plotting: pip install matplotlib")
