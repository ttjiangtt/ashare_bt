"""
signals/nine_turns.py
~~~~~~~~~~~~~~~~~~~~~
Implements the "Magical Nine Turns" (神奇九转) signal, based on
Tom DeMark's TD Sequential setup phase.

Rules
-----
At each bar i, compare close[i] to close[i-4]:

  Buy setup count:
    - Increment if close[i] < close[i-4]
    - Reset to 0 otherwise
    - Signal fires when count reaches exactly 9 (a "perfect 9")

  Sell setup count:
    - Increment if close[i] > close[i-4]
    - Reset to 0 otherwise
    - Signal fires when count reaches exactly 9

Optional perfection filter (--perfect=True):
    A "perfect" buy 9 requires that bar 8 or bar 9's low is <=
    the low of bar 6 or bar 7.  Filters out weaker setups.
    Similarly for sell: bar 8 or 9 high >= bar 6 or 7 high.

Classes
-------
  NineTurnsSignals   — computes setup counts and fires signals
  (Markout reused from williams_signals.py unchanged)

Usage
-----
    from signals.nine_turns import NineTurnsSignals
    from signals.williams_signals import Markout

    nt  = NineTurnsSignals(df, perfect=True).fit()
    mo  = Markout(nt, horizons=[1,2,3,5,10,20]).fit()

    print(nt.to_dataframe())
    print(mo.stats)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


# ── Signal dataclass (same shape as Williams Signal for Markout compatibility) ─

@dataclass
class Signal:
    bar:         int
    date:        pd.Timestamp
    direction:   int          # +1 buy, -1 sell
    entry_close: float
    trigger:     str
    it_trend:    int = 0      # not used, kept for Markout compatibility
    count:       int = 9      # always 9 for nine turns


# ── NineTurnsSignals ──────────────────────────────────────────────────────────

class NineTurnsSignals:
    """
    Detects Magical Nine Turns (神奇九转) setup signals.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with DatetimeIndex (from LocalDataAPI.get()).
    perfect : bool
        If True, apply the "perfection" filter:
          - Buy  9: low of bar 8 or 9 <= low  of bar 6 or 7
          - Sell 9: high of bar 8 or 9 >= high of bar 6 or 7
        Reduces signal count but improves quality.
        Default False.
    side : str | None
        ``'BUY'``, ``'SHORT'``, or None (both).

    After calling fit():
        .signals    — list of Signal objects
        .signal_df  — pd.DataFrame with count_buy, count_sell, signal columns
        .counts     — pd.DataFrame with full bar-by-bar counts (useful for plotting)
    """

    class _SwingsShim:
        """Minimal shim so Markout can access .swings.df without modification."""
        def __init__(self, df):
            self.df = df

    def __init__(
        self,
        df: pd.DataFrame,
        perfect: bool = False,
        side: Optional[str] = None,
    ) -> None:
        self._df     = df.copy()
        self.swings  = self._SwingsShim(self._df)   # Markout compatibility
        self.perfect = perfect
        self.side    = side.upper() if side else None
        self.signals: List[Signal] = []
        self.signal_df: Optional[pd.DataFrame] = None
        self.counts:    Optional[pd.DataFrame] = None
        self._fitted = False

    def fit(self) -> "NineTurnsSignals":
        df     = self._df
        closes = df["close"].values
        highs  = df["high"].values
        lows   = df["low"].values
        dates  = df.index
        n      = len(df)

        buy_count  = np.zeros(n, dtype=int)
        sell_count = np.zeros(n, dtype=int)

        # ── Compute bar-by-bar counts ─────────────────────────────────────
        for i in range(4, n):
            # Buy setup: close < close 4 bars ago
            if closes[i] < closes[i - 4]:
                buy_count[i] = buy_count[i - 1] + 1
            else:
                buy_count[i] = 0

            # Sell setup: close > close 4 bars ago
            if closes[i] > closes[i - 4]:
                sell_count[i] = sell_count[i - 1] + 1
            else:
                sell_count[i] = 0

        # ── Fire signals at count == 9 ────────────────────────────────────
        signals: List[Signal] = []

        for i in range(n):
            # BUY signal
            if buy_count[i] == 9:
                if self.side in (None, "BUY"):
                    if not self.perfect or self._check_perfect_buy(i, lows, buy_count):
                        signals.append(Signal(
                            bar=i, date=dates[i], direction=+1,
                            entry_close=float(closes[i]),
                            trigger=f"Nine Turns BUY{'(perfect)' if self.perfect else ''} @ {dates[i].date()}",
                            count=9,
                        ))

            # SELL signal
            if sell_count[i] == 9:
                if self.side in (None, "SHORT"):
                    if not self.perfect or self._check_perfect_sell(i, highs, sell_count):
                        signals.append(Signal(
                            bar=i, date=dates[i], direction=-1,
                            entry_close=float(closes[i]),
                            trigger=f"Nine Turns SELL{'(perfect)' if self.perfect else ''} @ {dates[i].date()}",
                            count=9,
                        ))

        self.signals = signals

        # ── Build aligned DataFrames ──────────────────────────────────────
        self.counts = pd.DataFrame({
            "buy_count":  buy_count,
            "sell_count": sell_count,
        }, index=df.index)

        sig_df = pd.DataFrame({
            "signal":      0,
            "entry_close": np.nan,
        }, index=df.index)

        for s in signals:
            sig_df.at[s.date, "signal"]      = s.direction
            sig_df.at[s.date, "entry_close"] = s.entry_close

        self.signal_df = sig_df
        self._fitted   = True
        return self

    # ── Perfection checks ─────────────────────────────────────────────────────

    @staticmethod
    def _check_perfect_buy(i: int, lows: np.ndarray, buy_count: np.ndarray) -> bool:
        """
        Perfect buy: low of bar 8 or bar 9 <= low of bar 6 or bar 7.
        Bar 9 is at index i; walk back to find bar 6, 7, 8.
        """
        # Find the bar indices for setup bars 6, 7, 8, 9
        # buy_count[i]==9 means bars i-8..i are the setup sequence
        b9 = i
        b8 = i - 1
        b7 = i - 2
        b6 = i - 3
        if b6 < 0:
            return False
        low_89  = min(lows[b8], lows[b9])
        low_67  = min(lows[b6], lows[b7])
        return low_89 <= low_67

    @staticmethod
    def _check_perfect_sell(i: int, highs: np.ndarray, sell_count: np.ndarray) -> bool:
        """
        Perfect sell: high of bar 8 or bar 9 >= high of bar 6 or bar 7.
        """
        b9 = i
        b8 = i - 1
        b7 = i - 2
        b6 = i - 3
        if b6 < 0:
            return False
        high_89 = max(highs[b8], highs[b9])
        high_67 = max(highs[b6], highs[b7])
        return high_89 >= high_67

    # ── Accessors ─────────────────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        if not self.signals:
            return pd.DataFrame()
        return pd.DataFrame([{
            "date":        s.date.date(),
            "direction":   s.direction,
            "side":        "BUY" if s.direction == 1 else "SHORT",
            "entry_close": round(s.entry_close, 3),
            "trigger":     s.trigger,
        } for s in self.signals])

    def summary(self) -> pd.Series:
        buys   = sum(1 for s in self.signals if s.direction ==  1)
        shorts = sum(1 for s in self.signals if s.direction == -1)
        return pd.Series({
            "total_signals": len(self.signals),
            "buy_signals":   buys,
            "short_signals": shorts,
            "perfect_filter": self.perfect,
        })

    @property
    def df(self) -> pd.DataFrame:
        """Price DataFrame — for Markout compatibility."""
        return self._df

    def __repr__(self) -> str:
        return (
            f"NineTurnsSignals(bars={len(self.df)}, "
            f"signals={len(self.signals)}, perfect={self.perfect})"
        )
