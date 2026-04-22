"""
Tests for ashare_bt.
Run with:  pytest tests/
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ashare_bt import Backtest, DataFeed, Strategy
from ashare_bt.strategy import SMACross, RSIStrategy, MACDStrategy, BollingerStrategy
from ashare_bt.utils.indicators import sma, ema, rsi, macd, bollinger, atr, kdj


# ─── Fixtures ────────────────────────────────────────────────────────────────────

def make_price_df(n=500, seed=42):
    """Generate a realistic OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 10.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high  = np.maximum(close, open_) * (1 + rng.uniform(0, 0.01, n))
    low   = np.minimum(close, open_) * (1 - rng.uniform(0, 0.01, n))
    vol   = rng.integers(500_000, 3_000_000, n)
    return pd.DataFrame({"date": dates, "open": open_, "high": high,
                          "low": low, "close": close, "volume": vol})


@pytest.fixture
def df():
    return make_price_df()


@pytest.fixture
def feed(df):
    return DataFeed(df, symbol="TEST")


# ─── Indicator tests ──────────────────────────────────────────────────────────────

class TestIndicators:
    def test_sma_length(self, df):
        result = sma(df["close"].values, 5)
        assert len(result) == len(df)
        assert np.isnan(result[3])
        assert not np.isnan(result[4])

    def test_ema_no_nan_after_start(self, df):
        result = ema(df["close"].values, 10)
        assert not np.isnan(result[0])
        assert np.all(np.isfinite(result))

    def test_rsi_bounds(self, df):
        result = rsi(df["close"].values, 14)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0)
        assert np.all(valid <= 100)

    def test_macd_returns_three_arrays(self, df):
        m, s, h = macd(df["close"].values)
        assert len(m) == len(s) == len(h) == len(df)

    def test_bollinger_upper_gt_lower(self, df):
        upper, mid, lower = bollinger(df["close"].values, 20, 2)
        valid = ~(np.isnan(upper) | np.isnan(lower))
        assert np.all(upper[valid] >= lower[valid])

    def test_kdj_output(self, df):
        K, D, J = kdj(df["high"].values, df["low"].values, df["close"].values)
        assert len(K) == len(df)


# ─── DataFeed tests ───────────────────────────────────────────────────────────────

class TestDataFeed:
    def test_load_dataframe(self, df):
        feed = DataFeed(df, symbol="X")
        assert len(feed) == len(df)

    def test_column_aliases(self):
        df = make_price_df(100)
        df = df.rename(columns={"close": "收盘", "volume": "成交量"})
        feed = DataFeed(df)
        assert "close" in feed.data.columns

    def test_cursor_history(self, feed):
        feed._cursor = 50
        hist = feed.history(10)
        assert len(hist) == 10

    def test_closes_array(self, feed):
        feed._cursor = 99
        c = feed.closes(20)
        assert len(c) == 20

    def test_missing_column_raises(self):
        df = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=5),
                            "close": [1, 2, 3, 4, 5]})
        with pytest.raises(ValueError):
            DataFeed(df)


# ─── Backtest engine tests ────────────────────────────────────────────────────────

class TestBacktest:
    def test_sma_cross_runs(self, df):
        bt = Backtest(df, SMACross, cash=100_000, fast=5, slow=20)
        result = bt.run()
        assert result.metrics["final_equity"] > 0

    def test_equity_curve_length(self, df):
        bt = Backtest(df, SMACross)
        result = bt.run()
        assert len(result.equity_curve) == len(df)

    def test_t1_enforced(self, df):
        """Shares bought on day N cannot be sold until day N+1."""
        bt = Backtest(df, SMACross, fast=5, slow=20)
        result = bt.run()
        for t in result.trades:
            assert t.exit_date > t.entry_date

    def test_lot_size(self, df):
        """All trades must be multiples of 100 shares."""
        bt = Backtest(df, SMACross, fast=5, slow=20)
        result = bt.run()
        for t in result.trades:
            assert t.shares % 100 == 0

    def test_no_negative_cash(self, df):
        """Broker should never go negative."""
        bt = Backtest(df, SMACross, cash=50_000, fast=5, slow=20)
        result = bt.run()
        assert result.metrics["final_equity"] >= 0

    def test_rsi_strategy(self, df):
        bt = Backtest(df, RSIStrategy, cash=100_000)
        result = bt.run()
        assert result.metrics is not None

    def test_macd_strategy(self, df):
        bt = Backtest(df, MACDStrategy, cash=100_000)
        result = bt.run()
        assert isinstance(result.trades_df(), pd.DataFrame)

    def test_bollinger_strategy(self, df):
        bt = Backtest(df, BollingerStrategy, cash=100_000)
        result = bt.run()
        assert result.metrics["n_trades"] >= 0

    def test_trades_df_columns(self, df):
        bt = Backtest(df, SMACross, fast=5, slow=20)
        result = bt.run()
        tdf = result.trades_df()
        if not tdf.empty:
            for col in ("entry_date", "exit_date", "pnl", "return_pct", "holding_days"):
                assert col in tdf.columns

    def test_drawdown_series_non_positive(self, df):
        bt = Backtest(df, SMACross)
        result = bt.run()
        dd = result.drawdown_series()
        assert np.all(dd.values <= 0.01)  # small tolerance for floating point

    def test_summary_returns_series(self, df):
        bt = Backtest(df, SMACross)
        result = bt.run()
        s = result.summary()
        assert isinstance(s, pd.Series)
        assert len(s) > 0


# ─── Optimisation tests ───────────────────────────────────────────────────────────

class TestOptimise:
    def test_optimise_returns_dataframe(self, df):
        bt = Backtest(df, SMACross, cash=100_000)
        grid = bt.optimise(fast=range(3, 8, 2), slow=range(15, 25, 5))
        assert isinstance(grid, pd.DataFrame)
        assert "sharpe" in grid.columns

    def test_optimise_skips_invalid_combos(self, df):
        """fast >= slow combos should be excluded."""
        bt = Backtest(df, SMACross)
        grid = bt.optimise(fast=[5, 20], slow=[10, 15])
        # fast=20, slow=10 should be dropped; fast=20, slow=15 also dropped
        if not grid.empty:
            assert (grid["fast"] < grid["slow"]).all()


# ─── Custom strategy test ─────────────────────────────────────────────────────────

class TestCustomStrategy:
    def test_custom_strategy(self, df):
        from ashare_bt.utils.indicators import sma as _sma

        class TripleMA(Strategy):
            fast = 5
            mid  = 10
            slow = 20

            def init(self):
                c = self.data.closes()
                self.f = self.indicator(_sma, c, self.fast)
                self.m = self.indicator(_sma, c, self.mid)
                self.s = self.indicator(_sma, c, self.slow)

            def next(self):
                i = self.data._cursor
                f = self.f[:i + 1]
                m = self.m[:i + 1]
                s = self.s[:i + 1]
                if self.position is None:
                    if self.crossover(f, m) and f[-1] > s[-1]:
                        self.buy(comment="triple_entry")
                else:
                    if self.crossunder(f, m):
                        self.sell(comment="triple_exit")

        bt = Backtest(df, TripleMA, cash=100_000)
        result = bt.run()
        assert result.metrics["final_equity"] > 0
