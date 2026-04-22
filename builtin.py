"""
ashare_bt.strategy.builtin
~~~~~~~~~~~~~~~~~~~~~~~~~~
Ready-to-use strategy implementations.
All class-level attributes act as **default parameters** that can be
overridden when constructing a :class:`~ashare_bt.engine.backtest.Backtest`::

    bt = Backtest(data, SMACross, fast=10, slow=30)
"""

from __future__ import annotations

import numpy as np

from base import Strategy, crossover, crossunder
from indicators import sma, ema, rsi as _rsi, macd as _macd, bollinger as _bollinger, kdj as _kdj


class SMACross(Strategy):
    """
    Dual Simple Moving Average crossover.

    Buy when the fast SMA crosses above the slow SMA,
    sell when it crosses below.

    Parameters
    ----------
    fast : int
        Period of the fast SMA.
    slow : int
        Period of the slow SMA.
    """

    fast: int = 5
    slow: int = 20

    def init(self) -> None:
        closes = self.data.closes()
        self.fast_ma = self.indicator(sma, closes, self.fast)
        self.slow_ma = self.indicator(sma, closes, self.slow)

    def next(self) -> None:
        i = self.data._cursor
        f = self.fast_ma[:i + 1]
        s = self.slow_ma[:i + 1]

        if self.position is None:
            if crossover(f, s):
                self.buy()
        else:
            if crossunder(f, s):
                self.sell()


class EMACross(Strategy):
    """
    Dual Exponential Moving Average crossover.

    Parameters
    ----------
    fast : int
        Fast EMA period.
    slow : int
        Slow EMA period.
    """

    fast: int = 12
    slow: int = 26

    def init(self) -> None:
        closes = self.data.closes()
        self.fast_ma = self.indicator(ema, closes, self.fast)
        self.slow_ma = self.indicator(ema, closes, self.slow)

    def next(self) -> None:
        i = self.data._cursor
        f = self.fast_ma[:i + 1]
        s = self.slow_ma[:i + 1]

        if self.position is None:
            if crossover(f, s):
                self.buy()
        else:
            if crossunder(f, s):
                self.sell()


class RSIStrategy(Strategy):
    """
    RSI mean-reversion strategy.

    Buy when RSI crosses above *oversold* from below.
    Sell when RSI crosses above *overbought*.

    Parameters
    ----------
    period : int
    oversold : float
    overbought : float
    """

    period: int = 14
    oversold: float = 30.0
    overbought: float = 70.0

    def init(self) -> None:
        self.rsi_arr = self.indicator(_rsi, self.data.closes(), self.period)

    def next(self) -> None:
        i = self.data._cursor
        r = self.rsi_arr[:i + 1]
        if len(r) < 2 or np.isnan(r[-1]) or np.isnan(r[-2]):
            return

        if self.position is None:
            if r[-2] < self.oversold <= r[-1]:
                self.buy()
        else:
            if r[-2] < self.overbought <= r[-1]:
                self.sell()


class MACDStrategy(Strategy):
    """
    Classic MACD crossover strategy.

    Buy when MACD line crosses above signal line,
    sell when it crosses below.

    Parameters
    ----------
    fast : int
    slow : int
    signal : int
    """

    fast: int = 12
    slow: int = 26
    signal: int = 9

    def init(self) -> None:
        closes = self.data.closes()
        macd_line, signal_line, hist = _macd(closes, self.fast, self.slow, self.signal)
        self.macd_line = self.indicator(lambda: macd_line)
        self.signal_line = self.indicator(lambda: signal_line)

    def next(self) -> None:
        i = self.data._cursor
        m = self.macd_line[:i + 1]
        s = self.signal_line[:i + 1]

        if self.position is None:
            if crossover(m, s):
                self.buy()
        else:
            if crossunder(m, s):
                self.sell()


class BollingerStrategy(Strategy):
    """
    Bollinger Band mean-reversion strategy.

    Buy when price touches the lower band, sell when it touches the upper band.

    Parameters
    ----------
    period : int
    num_std : float
    """

    period: int = 20
    num_std: float = 2.0

    def init(self) -> None:
        closes = self.data.closes()
        upper, mid, lower = _bollinger(closes, self.period, self.num_std)
        self.upper = self.indicator(lambda: upper)
        self.lower = self.indicator(lambda: lower)

    def next(self) -> None:
        i = self.data._cursor
        price = self.data.close
        upper = self.upper[i]
        lower = self.lower[i]

        if np.isnan(upper) or np.isnan(lower):
            return

        if self.position is None:
            if price <= lower:
                self.buy()
        else:
            if price >= upper:
                self.sell()


class KDJStrategy(Strategy):
    """
    KDJ-based strategy popular in Chinese retail trading.

    Buy when K crosses above D from below 20 (oversold).
    Sell when K crosses below D from above 80 (overbought).

    Parameters
    ----------
    n : int
        Lookback period for highest-high / lowest-low.
    m1 : int
        K smoothing.
    m2 : int
        D smoothing.
    oversold : float
    overbought : float
    """

    n: int = 9
    m1: int = 3
    m2: int = 3
    oversold: float = 20.0
    overbought: float = 80.0

    def init(self) -> None:
        h = self.data.data["high"].to_numpy(float)
        l = self.data.data["low"].to_numpy(float)
        c = self.data.closes()
        K, D, J = _kdj(h, l, c, self.n, self.m1, self.m2)
        self.K = self.indicator(lambda: K)
        self.D = self.indicator(lambda: D)

    def next(self) -> None:
        i = self.data._cursor
        k = self.K[:i + 1]
        d = self.D[:i + 1]

        if len(k) < 2 or np.isnan(k[-1]) or np.isnan(d[-1]):
            return

        if self.position is None:
            if k[-2] < d[-2] and k[-1] >= d[-1] and k[-1] < self.oversold:
                self.buy()
        else:
            if k[-2] > d[-2] and k[-1] <= d[-1] and k[-1] > self.overbought:
                self.sell()
