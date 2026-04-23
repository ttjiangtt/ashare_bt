"""
signals/williams_signals.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Implements Larry Williams's Chapter 1 entry signals from
"Long-Term Secrets to Short-Term Trading":

  1. WilliamsSwings  — identifies short-term and intermediate-term
                       swing highs/lows from raw OHLCV data.

  2. WilliamsSignals — generates buy/short entry signals based on
                       confirmed swing structure and trend direction.

  3. Markout         — given a signal series, computes forward P&L
                       at 1, 2, 3, 5, 10, 20 (or custom) days out,
                       assuming entry at the close of the signal bar
                       and exit at the close N days later.

Definitions (from the book)
---------------------------
Short-term high (STH):
    A bar whose high is strictly greater than both the prior bar's
    high and the next bar's high.  Inside days are skipped.
    Confirmed (locked in) when price subsequently falls below
    the low of the STH bar.

Short-term low (STL):
    A bar whose low is strictly less than both the prior bar's low
    and the next bar's low.
    Confirmed when price subsequently rallies above the high
    of the STL bar.

Intermediate-term high (ITH):
    A STH that has a lower STH on each side of it.

Intermediate-term low (ITL):
    A STL that has a higher STL on each side of it.

Trend direction:
    Up   — most recent confirmed ITL is higher than the one before it.
    Down — most recent confirmed ITH is lower than the one before it.

Buy signal:
    Trend is up AND a new STL is confirmed (price exceeds the high
    of the STL bar).  Entry at that bar's close.

Short signal:
    Trend is down AND a new STH is confirmed (price falls below the
    low of the STH bar).  Entry at that bar's close.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ─────────────────────────────────────────────────────────────────────────────
# 1.  WilliamsSwings
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SwingPoint:
    """A single identified swing point."""
    bar:       int            # integer position in the DataFrame
    date:      pd.Timestamp
    price:     float          # high for STH/ITH, low for STL/ITL
    kind:      str            # 'STH' | 'STL' | 'ITH' | 'ITL'
    confirmed: bool = False
    confirmed_bar: Optional[int] = None
    confirmed_date: Optional[pd.Timestamp] = None


class WilliamsSwings:
    """
    Identifies short-term and intermediate-term swing highs and lows.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with a DatetimeIndex and columns
        open, high, low, close.  Typically from LocalDataAPI.get().
    skip_inside : bool
        If True (default), inside bars (high ≤ prev high AND low ≥ prev low)
        are ignored when scanning for swing points, matching Williams's rule.

    After calling fit():
        .st_highs   — list of confirmed short-term highs
        .st_lows    — list of confirmed short-term lows
        .it_highs   — list of confirmed intermediate-term highs
        .it_lows    — list of confirmed intermediate-term lows
        .labels     — pd.DataFrame aligned to df with columns:
                        sth, stl, ith, itl  (bool flags per bar)

    Example
    -------
    >>> swings = WilliamsSwings(df).fit()
    >>> swings.labels[swings.labels["ith"]]["high"]
    """

    def __init__(self, df: pd.DataFrame, skip_inside: bool = True) -> None:
        self.df = df.copy()
        self.skip_inside = skip_inside
        self.st_highs:  List[SwingPoint] = []
        self.st_lows:   List[SwingPoint] = []
        self.it_highs:  List[SwingPoint] = []
        self.it_lows:   List[SwingPoint] = []
        self._fitted = False

    def fit(self) -> "WilliamsSwings":
        df = self.df
        highs = df["high"].values
        lows  = df["low"].values
        dates = df.index

        n = len(df)

        # ── Step 1: find non-inside bars ───────────────────────────────────
        # Inside bar: high <= prev high AND low >= prev low → skip
        non_inside = [0]   # always include first bar as anchor
        for i in range(1, n):
            if self.skip_inside:
                if highs[i] <= highs[i - 1] and lows[i] >= lows[i - 1]:
                    continue  # inside bar — skip
            non_inside.append(i)

        # ── Step 2: identify STH / STL from non-inside sequence ───────────
        # We need 3 consecutive non-inside bars: left, centre, right
        sth_candidates: List[Tuple[int, float]] = []   # (bar_idx, high)
        stl_candidates: List[Tuple[int, float]] = []   # (bar_idx, low)

        for j in range(1, len(non_inside) - 1):
            left   = non_inside[j - 1]
            centre = non_inside[j]
            right  = non_inside[j + 1]

            # STH: centre high > left high AND centre high > right high
            if highs[centre] > highs[left] and highs[centre] > highs[right]:
                sth_candidates.append((centre, highs[centre]))

            # STL: centre low < left low AND centre low < right low
            if lows[centre] < lows[left] and lows[centre] < lows[right]:
                stl_candidates.append((centre, lows[centre]))

        # ── Step 3: confirm STHs and STLs ─────────────────────────────────
        # STH confirmed when price subsequently falls below the low of the STH bar
        # STL confirmed when price subsequently rallies above the high of the STL bar

        confirmed_sth: List[SwingPoint] = []
        confirmed_stl: List[SwingPoint] = []

        for (bar_i, h) in sth_candidates:
            low_of_sth = lows[bar_i]
            for j in range(bar_i + 1, n):
                if lows[j] < low_of_sth:
                    sp = SwingPoint(
                        bar=bar_i, date=dates[bar_i], price=highs[bar_i],
                        kind="STH", confirmed=True,
                        confirmed_bar=j, confirmed_date=dates[j],
                    )
                    confirmed_sth.append(sp)
                    break

        for (bar_i, l) in stl_candidates:
            high_of_stl = highs[bar_i]
            for j in range(bar_i + 1, n):
                if highs[j] > high_of_stl:
                    sp = SwingPoint(
                        bar=bar_i, date=dates[bar_i], price=lows[bar_i],
                        kind="STL", confirmed=True,
                        confirmed_bar=j, confirmed_date=dates[j],
                    )
                    confirmed_stl.append(sp)
                    break

        self.st_highs = confirmed_sth
        self.st_lows  = confirmed_stl

        # ── Step 4: identify ITHs and ITLs ────────────────────────────────
        # ITH: a STH with a lower STH on each side
        # ITL: a STL with a higher STL on each side
        confirmed_ith: List[SwingPoint] = []
        confirmed_itl: List[SwingPoint] = []

        for j in range(1, len(confirmed_sth) - 1):
            left   = confirmed_sth[j - 1]
            centre = confirmed_sth[j]
            right  = confirmed_sth[j + 1]
            if centre.price > left.price and centre.price > right.price:
                sp = SwingPoint(
                    bar=centre.bar, date=centre.date, price=centre.price,
                    kind="ITH", confirmed=True,
                    confirmed_bar=centre.confirmed_bar,
                    confirmed_date=centre.confirmed_date,
                )
                confirmed_ith.append(sp)

        for j in range(1, len(confirmed_stl) - 1):
            left   = confirmed_stl[j - 1]
            centre = confirmed_stl[j]
            right  = confirmed_stl[j + 1]
            if centre.price < left.price and centre.price < right.price:
                sp = SwingPoint(
                    bar=centre.bar, date=centre.date, price=centre.price,
                    kind="ITL", confirmed=True,
                    confirmed_bar=centre.confirmed_bar,
                    confirmed_date=centre.confirmed_date,
                )
                confirmed_itl.append(sp)

        self.it_highs = confirmed_ith
        self.it_lows  = confirmed_itl

        # ── Step 5: build label DataFrame ─────────────────────────────────
        labels = pd.DataFrame(
            {"sth": False, "stl": False, "ith": False, "itl": False},
            index=df.index,
        )
        for sp in confirmed_sth:
            labels.at[sp.date, "sth"] = True
        for sp in confirmed_stl:
            labels.at[sp.date, "stl"] = True
        for sp in confirmed_ith:
            labels.at[sp.date, "ith"] = True
        for sp in confirmed_itl:
            labels.at[sp.date, "itl"] = True

        self.labels = labels
        self._fitted = True
        return self

    def summary(self) -> pd.DataFrame:
        """Return a tidy DataFrame of all swing points."""
        rows = (
            self.st_highs + self.st_lows +
            self.it_highs + self.it_lows
        )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([{
            "date":           sp.date.date(),
            "kind":           sp.kind,
            "price":          round(sp.price, 3),
            "confirmed_date": sp.confirmed_date.date() if sp.confirmed_date else None,
        } for sp in rows]).sort_values("date").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  WilliamsSignals
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Signal:
    """A single entry signal."""
    bar:        int
    date:       pd.Timestamp
    direction:  int            # +1 = buy, -1 = short
    entry_close: float         # close price at signal bar (entry price)
    trigger:    str            # description e.g. "STL confirmed in uptrend"
    it_trend:   int            # +1 up, -1 down, 0 neutral at signal time


class WilliamsSignals:
    """
    Generates buy and short-sell entry signals from confirmed swing structure.

    Signal logic (strict Williams Chapter 1):
    - Buy:   trend is UP (last ITL > prior ITL)  AND  a new STL is just confirmed.
             Entry at the close of the confirmation bar.
    - Short: trend is DOWN (last ITH < prior ITH) AND  a new STH is just confirmed.
             Entry at the close of the confirmation bar.

    Parameters
    ----------
    swings : WilliamsSwings
        A fitted WilliamsSwings instance.
    require_it_trend : bool
        If True (default), only signal when intermediate-term trend
        is confirmed (requires at least 2 ITL/ITH to establish direction).
        If False, generates signals on every confirmed STL/STH regardless
        of intermediate trend.
    avoid_prev_opposite : bool
        Controls the "recent opposite signal" filter.

        False (default) — no filter; generate the signal regardless of
              what happened in the lookback window.  This is the original
              behaviour.

        True  — stronger filter: suppress a signal if an *opposite* signal
              was generated within the previous `lookback` bars.
              Example: a BUY signal is skipped if a SHORT fired ≤ 3 bars
              ago, because the market has recently shown the opposite
              conviction and may still be in that move.
    lookback : int
        Number of bars to look back for the opposite-signal filter.
        Only used when avoid_prev_opposite=False.  Default 3.

    After calling fit():
        .signals   — list of Signal objects (after all filters applied)
        .signal_df — pd.DataFrame aligned to price data
        .n_suppressed — int, signals dropped by the avoid_prev_opposite filter
    """

    def __init__(
        self,
        swings: WilliamsSwings,
        require_it_trend: bool = True,
        avoid_prev_opposite: bool = False,
        lookback: int = 3,
    ) -> None:
        if not swings._fitted:
            raise ValueError("Call WilliamsSwings.fit() before passing to WilliamsSignals.")
        self.swings              = swings
        self.require_it_trend    = require_it_trend
        self.avoid_prev_opposite = avoid_prev_opposite
        self.lookback            = lookback
        self.signals:      List[Signal] = []
        self.signal_df:    Optional[pd.DataFrame] = None
        self.n_suppressed: int = 0

    def fit(self) -> "WilliamsSignals":
        df     = self.swings.df
        closes = df["close"].values
        dates  = df.index
        n      = len(df)

        stl_list = self.swings.st_lows
        sth_list = self.swings.st_highs
        itl_list = self.swings.it_lows
        ith_list = self.swings.it_highs

        signals: List[Signal] = []

        # ── Buy signals: confirmed STL in uptrend ─────────────────────────
        for stl in stl_list:
            conf_bar = stl.confirmed_bar
            if conf_bar is None or conf_bar >= n:
                continue

            trend = self._it_trend_at(conf_bar, itl_list, ith_list)

            if self.require_it_trend and trend != 1:
                continue

            signals.append(Signal(
                bar=conf_bar,
                date=dates[conf_bar],
                direction=+1,
                entry_close=float(closes[conf_bar]),
                trigger=f"STL @ {stl.date.date()} confirmed, IT trend={trend:+d}",
                it_trend=trend,
            ))

        # ── Short signals: confirmed STH in downtrend ─────────────────────
        for sth in sth_list:
            conf_bar = sth.confirmed_bar
            if conf_bar is None or conf_bar >= n:
                continue

            trend = self._it_trend_at(conf_bar, itl_list, ith_list)

            if self.require_it_trend and trend != -1:
                continue

            signals.append(Signal(
                bar=conf_bar,
                date=dates[conf_bar],
                direction=-1,
                entry_close=float(closes[conf_bar]),
                trigger=f"STH @ {sth.date.date()} confirmed, IT trend={trend:+d}",
                it_trend=trend,
            ))

        # Sort chronologically before applying the lookback filter
        signals.sort(key=lambda s: s.bar)

        # ── avoid_prev_opposite filter ────────────────────────────────────
        # When avoid_prev_opposite=False (the stronger filter):
        #   walk through signals in order; keep a running record of the
        #   most recent signal bar per direction.  Before accepting a new
        #   signal, check whether an opposite signal fired within the last
        #   `lookback` bars.  If so, suppress it.
        if self.avoid_prev_opposite:
            filtered    = []
            last_bar_by_dir: dict[int, int] = {}   # direction → last bar index
            n_suppressed = 0

            for sig in signals:
                opposite = -sig.direction
                last_opp = last_bar_by_dir.get(opposite, -9999)

                if sig.bar - last_opp <= self.lookback:
                    # Opposite signal was within lookback window — suppress
                    n_suppressed += 1
                else:
                    filtered.append(sig)

                # Always update the tracker for this direction, even if
                # suppressed, so the lookback stays accurate
                last_bar_by_dir[sig.direction] = sig.bar

            self.signals      = filtered
            self.n_suppressed = n_suppressed
        else:
            # avoid_prev_opposite=False — no filter, pass all signals through
            self.signals      = signals
            self.n_suppressed = 0

        # ── Build aligned DataFrame ───────────────────────────────────────
        sig_df = pd.DataFrame(
            {"signal": 0, "entry_close": np.nan},
            index=df.index,
        )
        for s in self.signals:
            sig_df.at[s.date, "signal"]      = s.direction
            sig_df.at[s.date, "entry_close"] = s.entry_close

        self.signal_df = sig_df
        return self

    @staticmethod
    def _it_trend_at(
        bar: int,
        itl_list: List[SwingPoint],
        ith_list: List[SwingPoint],
    ) -> int:
        """
        Determine the intermediate-term trend direction as of a given bar,
        using only ITLs/ITHs whose confirmation bar is <= bar (no look-ahead).

        Returns +1 (up), -1 (down), or 0 (indeterminate).
        """
        # ITLs known by this bar
        known_itl = [p for p in itl_list if p.confirmed_bar is not None
                     and p.confirmed_bar <= bar]
        # ITHs known by this bar
        known_ith = [p for p in ith_list if p.confirmed_bar is not None
                     and p.confirmed_bar <= bar]

        if len(known_itl) >= 2:
            if known_itl[-1].price > known_itl[-2].price:
                return +1

        if len(known_ith) >= 2:
            if known_ith[-1].price < known_ith[-2].price:
                return -1

        return 0

    def to_dataframe(self) -> pd.DataFrame:
        """Return signals as a clean DataFrame."""
        if not self.signals:
            return pd.DataFrame()
        rows = [{
            "date":        s.date.date(),
            "direction":   s.direction,
            "side":        "BUY" if s.direction == 1 else "SHORT",
            "entry_close": round(s.entry_close, 3),
            "it_trend":    s.it_trend,
            "trigger":     s.trigger,
        } for s in self.signals]
        return pd.DataFrame(rows)

    def summary(self) -> pd.Series:
        """Quick stats on signal counts."""
        buys   = sum(1 for s in self.signals if s.direction == 1)
        shorts = sum(1 for s in self.signals if s.direction == -1)
        return pd.Series({
            "total_signals":       len(self.signals),
            "buy_signals":         buys,
            "short_signals":       shorts,
            "suppressed":          self.n_suppressed,
            "avoid_prev_opposite": self.avoid_prev_opposite,
            "lookback":            self.lookback if self.avoid_prev_opposite else "n/a",
        })


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Markout
# ─────────────────────────────────────────────────────────────────────────────

class Markout:
    """
    Computes N-day forward P&L for each signal.

    Entry:  close of the signal bar.
    Exit:   depends on the `raw` flag:

        raw=True  (default)
            Hold exactly N days regardless of what happens in between.
            Exit at close of bar (entry_bar + N).

        raw=False
            Hold up to N days, but exit early at the close of the first
            bar where an *opposite* signal fires.  For a BUY, that means
            the first subsequent SHORT signal; for a SHORT, the first
            subsequent BUY signal.  If no opposite signal arrives within
            N days, exit at the horizon as normal.
            Each row in .raw gains two extra columns per horizon:
                exit_reason_{N}d  — "horizon" or "signal"
                days_held_{N}d    — actual trading days held

    P&L in both modes:
        direction × (exit_close / entry_close - 1) × 100  (%)

    Parameters
    ----------
    signals_obj : WilliamsSignals
        A fitted WilliamsSignals instance.
    horizons : list of int
        Forward horizons in trading days.  Default [1, 2, 3, 5, 10, 20].
    raw : bool
        True  → fixed N-day hold (original behaviour).
        False → early exit on opposite signal.

    After calling fit():
        .raw_df   — pd.DataFrame with one row per signal
        .stats    — pd.DataFrame with summary statistics per horizon
    """

    DEFAULT_HORIZONS = [1, 2, 3, 5, 10, 20]

    def __init__(
        self,
        signals_obj: WilliamsSignals,
        horizons: Optional[List[int]] = None,
        raw: bool = True,
        side: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        side : str or None
            Filter signals by direction before computing markout.
            ``'BUY'``   — long signals only.
            ``'SHORT'`` — short signals only.
            ``None``    — all signals (default).
        """
        if not signals_obj.signals:
            raise ValueError("No signals found — run WilliamsSignals.fit() first.")
        if side is not None and side.upper() not in ("BUY", "SHORT"):
            raise ValueError(f"side must be 'BUY', 'SHORT', or None — got {side!r}")
        self.signals_obj = signals_obj
        self.horizons    = horizons or self.DEFAULT_HORIZONS
        self.raw         = raw
        self.side        = side.upper() if side else None
        self.raw_df: Optional[pd.DataFrame] = None
        self.stats:  Optional[pd.DataFrame] = None

    # ── internal: build a lookup of {bar: direction} for all signals ──────
    def _opposite_signal_map(self) -> dict[int, int]:
        """Return {bar_index: direction} for every signal."""
        return {s.bar: s.direction for s in self.signals_obj.signals}

    def _first_opposite_bar(
        self,
        entry_bar: int,
        direction: int,
        horizon: int,
        sig_map: dict[int, int],
        n_bars: int,
    ) -> tuple[int, str]:
        """
        Scan forward from entry_bar+1 up to entry_bar+horizon.
        Return (exit_bar, exit_reason) where exit_reason is
        'signal' if an opposite signal triggered, else 'horizon'.
        """
        limit = min(entry_bar + horizon, n_bars - 1)
        for b in range(entry_bar + 1, limit + 1):
            if sig_map.get(b, 0) == -direction:   # opposite direction
                return b, "signal"
        return entry_bar + horizon, "horizon"

    def fit(self) -> "Markout":
        df     = self.signals_obj.swings.df
        closes = df["close"].values
        n      = len(closes)

        # Apply side filter
        signals_to_use = [
            s for s in self.signals_obj.signals
            if self.side is None
            or (self.side == "BUY"   and s.direction == 1)
            or (self.side == "SHORT" and s.direction == -1)
        ]
        if not signals_to_use:
            raise ValueError(
                f"No {self.side or ''} signals found after side filter."
            )

        sig_map = self._opposite_signal_map() if not self.raw else {}
        rows    = []

        for sig in signals_to_use:
            entry_bar = sig.bar
            if entry_bar >= n:
                continue

            entry_price = sig.entry_close
            row = {
                "date":        sig.date,
                "direction":   sig.direction,
                "side":        "BUY" if sig.direction == 1 else "SHORT",
                "entry_close": round(entry_price, 3),
                "it_trend":    sig.it_trend,
            }

            for h in self.horizons:
                if self.raw:
                    # ── fixed hold ────────────────────────────────────────
                    exit_bar    = entry_bar + h
                    exit_reason = "horizon"
                else:
                    # ── early-exit on opposite signal ─────────────────────
                    exit_bar, exit_reason = self._first_opposite_bar(
                        entry_bar, sig.direction, h, sig_map, n
                    )

                if exit_bar >= n:
                    row[f"ret_{h}d"] = np.nan
                    if not self.raw:
                        row[f"exit_reason_{h}d"] = np.nan
                        row[f"days_held_{h}d"]   = np.nan
                else:
                    exit_price  = closes[exit_bar]
                    raw_ret     = (exit_price / entry_price - 1) * 100
                    row[f"ret_{h}d"] = round(sig.direction * raw_ret, 4)
                    if not self.raw:
                        row[f"exit_reason_{h}d"] = exit_reason
                        row[f"days_held_{h}d"]   = exit_bar - entry_bar

            rows.append(row)

        self.raw_df = pd.DataFrame(rows).set_index("date")

        # ── Summary statistics per horizon ────────────────────────────────
        ret_cols  = [f"ret_{h}d" for h in self.horizons]
        stat_rows = []

        for col in ret_cols:
            series = self.raw_df[col].dropna()
            if len(series) < 2:
                continue
            wins = (series > 0).sum()
            t_stat, p_val = scipy_stats.ttest_1samp(series, 0)
            stat_row = {
                "horizon":  col,
                "n":        len(series),
                "mean_%":   round(series.mean(), 4),
                "median_%": round(series.median(), 4),
                "std_%":    round(series.std(), 4),
                "win_rate": round(wins / len(series), 4),
                "sharpe":   round(series.mean() / series.std(), 4) if series.std() > 0 else np.nan,
                "t_stat":   round(t_stat, 3),
                "p_value":  round(p_val, 4),
                "min_%":    round(series.min(), 4),
                "max_%":    round(series.max(), 4),
            }
            # signal-exit mode: add early-exit breakdown
            if not self.raw:
                reason_col = f"exit_reason_{col[4:]}"
                if reason_col in self.raw_df.columns:
                    reasons = self.raw_df[reason_col].dropna()
                    stat_row["early_exit_%"] = round(
                        (reasons == "signal").mean(), 4
                    )
                    avg_hold_col = f"days_held_{col[4:]}"
                    if avg_hold_col in self.raw_df.columns:
                        stat_row["avg_days_held"] = round(
                            self.raw_df[avg_hold_col].dropna().mean(), 2
                        )
            stat_rows.append(stat_row)

        self.stats = pd.DataFrame(stat_rows).set_index("horizon")
        return self

    def by_side(self) -> pd.DataFrame:
        """
        Summary statistics split by BUY vs SHORT signals.
        Returns a DataFrame indexed by (side, horizon).
        """
        ret_cols = [f"ret_{h}d" for h in self.horizons]
        rows = []
        for side in ("BUY", "SHORT"):
            subset = self.raw_df[self.raw_df["side"] == side]
            if subset.empty:
                continue
            for col in ret_cols:
                s = subset[col].dropna()
                if len(s) < 2:
                    continue
                t_stat, p_val = scipy_stats.ttest_1samp(s, 0)
                rows.append({
                    "side":     side,
                    "horizon":  col,
                    "n":        len(s),
                    "mean_%":   round(s.mean(), 4),
                    "win_rate": round((s > 0).mean(), 4),
                    "t_stat":   round(t_stat, 3),
                    "p_value":  round(p_val, 4),
                })
        return pd.DataFrame(rows).set_index(["side", "horizon"])

    def plot(self, figsize: tuple = (12, 5)) -> None:
        """
        Plot mean return and win rate across horizons, split by side.
        Requires matplotlib.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("pip install matplotlib")

        fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor="#0a0e17")
        ret_cols = [f"ret_{h}d" for h in self.horizons]
        horizon_labels = [str(h) for h in self.horizons]

        colors = {"BUY": "#00d68f", "SHORT": "#ff4d6d", "ALL": "#f0b429"}

        for ax in axes:
            ax.set_facecolor("#0d1420")
            ax.tick_params(colors="#6b8299")
            ax.spines[:].set_edgecolor("#1e2d45")
            ax.grid(color="#1e2d45", linewidth=0.5, linestyle="--")

        # Mean return
        ax = axes[0]
        for side, color in colors.items():
            if side == "ALL":
                s = self.raw_df
            else:
                s = self.raw_df[self.raw_df["side"] == side]
            if s.empty:
                continue
            means = [s[c].mean() for c in ret_cols]
            ax.plot(horizon_labels, means, marker="o", label=side, color=color, linewidth=1.8)
        ax.axhline(0, color="#3a5a7a", linewidth=0.8)
        ax.set_title("Mean Return % by Horizon", color="#c8d8e8", fontsize=11)
        ax.set_xlabel("Days", color="#6b8299")
        ax.set_ylabel("Mean Return %", color="#6b8299")
        ax.legend(facecolor="#0d1420", edgecolor="#1e2d45", labelcolor="#c8d8e8")

        # Win rate
        ax = axes[1]
        for side, color in colors.items():
            if side == "ALL":
                s = self.raw_df
            else:
                s = self.raw_df[self.raw_df["side"] == side]
            if s.empty:
                continue
            wr = [(s[c] > 0).mean() for c in ret_cols]
            ax.plot(horizon_labels, wr, marker="o", label=side, color=color, linewidth=1.8)
        ax.axhline(0.5, color="#3a5a7a", linewidth=0.8, linestyle="--")
        ax.set_title("Win Rate by Horizon", color="#c8d8e8", fontsize=11)
        ax.set_xlabel("Days", color="#6b8299")
        ax.set_ylabel("Win Rate", color="#6b8299")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.legend(facecolor="#0d1420", edgecolor="#1e2d45", labelcolor="#c8d8e8")

        plt.suptitle(
            f"Williams Swing Signals — Markout Analysis\n"
            f"({self.raw_df.index[0].date()} → {self.raw_df.index[-1].date()}, "
            f"n={len(self.raw_df)} signals)",
            color="#ddeeff", fontsize=12, y=1.01,
        )
        plt.tight_layout()
        plt.show()
