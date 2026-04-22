"""
ashare_bt.data.feed
~~~~~~~~~~~~~~~~~~~
Data ingestion layer.  Accepts a pandas DataFrame or CSV path and
normalises column names so the rest of the library always works with
a consistent schema.

Required columns (case-insensitive):
    date, open, high, low, close, volume

Optional:
    amount   (turnover in yuan)
    adj_factor

The feed keeps a read-only view of the raw data and provides
a rolling *window* during a backtest so strategies can look back
at history without peeking into the future.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import warnings

import numpy as np
import pandas as pd


# ── Column aliases ──────────────────────────────────────────────────────────────
_ALIASES: dict[str, list[str]] = {
    "date":   ["date", "trade_date", "tradedate", "datetime", "时间", "日期"],
    "open":   ["open", "开盘", "开盘价", "open_price"],
    "high":   ["high", "最高", "最高价", "high_price"],
    "low":    ["low", "最低", "最低价", "low_price"],
    "close":  ["close", "收盘", "收盘价", "close_price"],
    "volume": ["volume", "成交量", "vol"],
    "amount": ["amount", "成交额", "turnover"],
}

_REQUIRED = {"date", "open", "high", "low", "close", "volume"}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename aliased columns to canonical names."""
    rename_map: dict[str, str] = {}
    lower_cols = {c.lower(): c for c in df.columns}
    for canonical, aliases in _ALIASES.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            if alias.lower() in lower_cols:
                rename_map[lower_cols[alias.lower()]] = canonical
                break
    return df.rename(columns=rename_map)


class DataFeed:
    """
    Wrapper around a price DataFrame that provides safe, look-ahead-free
    access to OHLCV data.

    Parameters
    ----------
    source : str | Path | pd.DataFrame
        CSV file path or pre-built DataFrame.
    symbol : str, optional
        Ticker label (cosmetic).
    adjust : bool
        If *True* and an ``adj_factor`` column exists, multiply OHLC
        prices by the adjustment factor.
    freq : str
        Pandas offset alias for the data frequency.  Used for
        annualisation in metrics (e.g. ``'D'``, ``'W'``, ``'M'``).

    Examples
    --------
    >>> feed = DataFeed("600519.csv", symbol="600519")
    >>> feed = DataFeed(df, symbol="000001")
    """

    def __init__(
        self,
        source: Union[str, Path, pd.DataFrame],
        symbol: str = "UNKNOWN",
        adjust: bool = False,
        freq: str = "D",
    ) -> None:
        self.symbol = symbol
        self.freq = freq
        self._raw = self._load(source)
        self._data = self._prepare(self._raw, adjust)
        # Backtest engine sets this pointer during replay
        self._cursor: int = len(self._data) - 1

    # ── loading ─────────────────────────────────────────────────────────────────

    def _load(self, source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        if isinstance(source, (str, Path)):
            df = pd.read_csv(source)
        elif isinstance(source, pd.DataFrame):
            df = source.copy()
        else:
            raise TypeError(f"source must be str, Path or DataFrame, got {type(source)}")
        return df

    def _prepare(self, df: pd.DataFrame, adjust: bool) -> pd.DataFrame:
        df = _normalise_columns(df)

        missing = _REQUIRED - set(df.columns)
        if missing:
            raise ValueError(f"DataFeed missing required columns: {missing}")

        # Parse date
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Optional adjustment
        if adjust and "adj_factor" in df.columns:
            for col in ("open", "high", "low", "close"):
                df[col] = df[col] * df["adj_factor"]

        # Ensure numeric OHLCV
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        n_bad = df[["open", "high", "low", "close", "volume"]].isna().any(axis=1).sum()
        if n_bad > 0:
            warnings.warn(f"DataFeed '{self.symbol}': {n_bad} rows with NaN OHLCV — forward-filled.")
            df[["open", "high", "low", "close", "volume"]] = (
                df[["open", "high", "low", "close", "volume"]].ffill()
            )

        df = df.set_index("date")
        return df

    # ── public accessors ────────────────────────────────────────────────────────

    @property
    def data(self) -> pd.DataFrame:
        """Full OHLCV DataFrame (read-only view)."""
        return self._data

    def __len__(self) -> int:
        return len(self._data)

    @property
    def index(self) -> pd.DatetimeIndex:
        return self._data.index

    # ── windowed access used by strategies ──────────────────────────────────────

    def history(self, n: Optional[int] = None) -> pd.DataFrame:
        """
        Return the last *n* bars up to and including the current bar.
        If *n* is None, return all history to date.
        """
        end = self._cursor + 1
        if n is None:
            return self._data.iloc[:end]
        return self._data.iloc[max(0, end - n):end]

    @property
    def current(self) -> pd.Series:
        """The bar at the current cursor position."""
        return self._data.iloc[self._cursor]

    @property
    def close(self) -> float:
        return float(self._data.iloc[self._cursor]["close"])

    @property
    def open(self) -> float:
        return float(self._data.iloc[self._cursor]["open"])

    @property
    def high(self) -> float:
        return float(self._data.iloc[self._cursor]["high"])

    @property
    def low(self) -> float:
        return float(self._data.iloc[self._cursor]["low"])

    @property
    def volume(self) -> float:
        return float(self._data.iloc[self._cursor]["volume"])

    @property
    def date(self) -> pd.Timestamp:
        return self._data.index[self._cursor]

    def closes(self, n: Optional[int] = None) -> np.ndarray:
        """Convenience: array of recent close prices."""
        return self.history(n)["close"].to_numpy(dtype=float)

    def __repr__(self) -> str:
        return (
            f"DataFeed(symbol={self.symbol!r}, "
            f"bars={len(self._data)}, "
            f"range={self._data.index[0].date()}→{self._data.index[-1].date()})"
        )
