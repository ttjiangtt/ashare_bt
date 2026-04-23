"""
data/local_api.py
~~~~~~~~~~~~~~~~~
Local data API for reading A-share daily data from CSV files downloaded
by download_all.py, plus ticker/name lookup utilities.

All functions are stateless and work directly from your market_data folder.

Quick start
-----------
    from data.local_api import LocalDataAPI

    api = LocalDataAPI("C:/Users/ttjia/OneDrive/Work/ashare/market_data")

    # Price data
    df = api.get("600519")
    df = api.get("600519", start="2023-01-01", end="2023-12-31")
    df = api.get("贵州茅台")                    # name works too

    # Lookup
    api.name("600519")                          # → "贵州茅台"
    api.ticker("贵州茅台")                      # → "600519"
    api.search("茅台")                          # → DataFrame of matches
    api.info("600519")                          # → dict of metadata

    # Universe
    api.list_tickers()                          # → ["000001", "000002", ...]
    api.list_all()                              # → DataFrame with ticker + name
"""

from __future__ import annotations

import logging
import datetime as dt
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

import pandas as pd

log = logging.getLogger(__name__)

# ── Name registry ─────────────────────────────────────────────────────────────
# Embedded snapshot of common stocks so basic lookups work offline without
# needing to call AKShare. Full registry is built from your local files.

_COMMON_NAMES = {
    "000001": "平安银行",   "000002": "万科A",      "000858": "五粮液",
    "002415": "海康威视",   "000333": "美的集团",    "002594": "比亚迪",
    "000651": "格力电器",   "000725": "京东方A",     "000776": "广发证券",
    "600000": "浦发银行",   "600016": "民生银行",    "600019": "宝钢股份",
    "600028": "中国石化",   "600030": "中信证券",    "600036": "招商银行",
    "600048": "保利发展",   "600050": "中国联通",    "600104": "上汽集团",
    "600276": "恒瑞医药",   "600309": "万华化学",    "600519": "贵州茅台",
    "600585": "海螺水泥",   "600690": "海尔智家",    "600900": "长江电力",
    "601012": "隆基绿能",   "601088": "中国神华",    "601166": "兴业银行",
    "601288": "农业银行",   "601318": "中国平安",    "601398": "工商银行",
    "601601": "中国太保",   "601628": "中国人寿",    "601688": "华泰证券",
    "601857": "中国石油",   "601888": "中国中免",    "601899": "紫金矿业",
    "601939": "建设银行",   "601988": "中国银行",    "603259": "药明康德",
    "688981": "中芯国际",
}


class LocalDataAPI:
    """
    Read A-share daily OHLCV data from a local folder of CSV files.

    Parameters
    ----------
    root : str | Path
        Folder containing CSV files named ``{ticker}.csv``.
    name_file : str, optional
        Path to a CSV with columns ``ticker,name`` for name lookups.
        If not provided, falls back to the embedded common-name list
        and attempts to load from AKShare as a last resort.

    Examples
    --------
    >>> api = LocalDataAPI("C:/market_data")
    >>> df  = api.get("600519", start="2022-01-01")
    >>> api.name("600519")
    '贵州茅台'
    >>> api.ticker("茅台")
    '600519'
    """

    def __init__(
        self,
        root: Union[str, Path],
        name_file: Optional[Union[str, Path]] = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Data folder not found: {self.root}")

        # Build ticker→name and name→ticker maps
        self._t2n: dict[str, str] = {}   # ticker → name
        self._n2t: dict[str, str] = {}   # lowercase name → ticker

        self._load_names(name_file)

    # ── Name registry ─────────────────────────────────────────────────────────

    def _load_names(self, name_file: Optional[Path]) -> None:
        """Populate the name registry from multiple sources."""

        # 1. Embedded common names
        self._t2n.update(_COMMON_NAMES)

        # 2. User-supplied name file
        if name_file is not None:
            p = Path(name_file)
            if p.exists():
                df = pd.read_csv(p, dtype=str)
                df.columns = df.columns.str.lower()
                if {"ticker", "name"}.issubset(df.columns):
                    for _, row in df.iterrows():
                        t = str(row["ticker"]).zfill(6)
                        self._t2n[t] = row["name"]

        # 3. _names.csv auto-saved in the data root by download_all.py
        auto = self.root / "_names.csv"
        if auto.exists():
            try:
                df = pd.read_csv(auto, dtype=str)
                for _, row in df.iterrows():
                    t = str(row["ticker"]).zfill(6)
                    self._t2n[t] = row["name"]
            except Exception:
                pass

        # 4. Try AKShare (silent fail — not required)
        if len(self._t2n) < 100:
            self._try_load_from_akshare()

        # Build reverse map (lowercase for fuzzy matching)
        self._n2t = {v.lower(): k for k, v in self._t2n.items()}

    def _try_load_from_akshare(self) -> None:
        try:
            import akshare as ak
            df = ak.stock_info_a_code_name()
            for _, row in df.iterrows():
                t = str(row["code"]).zfill(6)
                self._t2n[t] = row["name"]
            # Cache it for next time
            self._save_names_cache()
        except Exception:
            pass  # AKShare unavailable — use embedded list

    def _save_names_cache(self) -> None:
        """Save ticker→name to _names.csv in the data root."""
        try:
            df = pd.DataFrame(
                [{"ticker": k, "name": v} for k, v in self._t2n.items()]
            )
            df.to_csv(self.root / "_names.csv", index=False)
        except Exception:
            pass

    # ── Core data access ──────────────────────────────────────────────────────

    def get(
        self,
        symbol: str,
        start: Optional[str] = None,
        end:   Optional[str] = None,
        columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Load OHLCV data for a ticker (or stock name).

        Parameters
        ----------
        symbol : str
            6-digit ticker OR Chinese stock name (e.g. "贵州茅台").
        start : str, optional  "YYYY-MM-DD"
        end   : str, optional  "YYYY-MM-DD"
        columns : list, optional
            Subset of columns to return, e.g. ["date","close","volume"].

        Returns
        -------
        pd.DataFrame  indexed by date, columns: open high low close volume [amount]

        Raises
        ------
        FileNotFoundError  if no CSV exists for the ticker.
        """
        ticker = self._resolve(symbol)
        path   = self.root / f"{ticker}.csv"

        if not path.exists():
            raise FileNotFoundError(
                f"No data file for {ticker} ({symbol}). "
                f"Run download_all.py to fetch it."
            )

        df = pd.read_csv(path, parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df = df.set_index("date")

        if start:
            df = df[df.index >= pd.Timestamp(start)]
        if end:
            df = df[df.index <= pd.Timestamp(end)]

        if columns:
            available = [c for c in columns if c in df.columns]
            df = df[available]

        return df

    def get_close(
        self,
        symbol: str,
        start: Optional[str] = None,
        end:   Optional[str] = None,
    ) -> pd.Series:
        """Return just the close price series."""
        return self.get(symbol, start=start, end=end)["close"].rename(symbol)

    def get_multi(
        self,
        symbols: list[str],
        field: str = "close",
        start: Optional[str] = None,
        end:   Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load a single field (e.g. close) for multiple tickers into a
        wide DataFrame with tickers as columns.

        Parameters
        ----------
        symbols : list of str   Tickers or names.
        field   : str           Column to extract (default "close").

        Returns
        -------
        pd.DataFrame  shape (dates × tickers)
        """
        series = {}
        for sym in symbols:
            try:
                ticker = self._resolve(sym)
                series[ticker] = self.get(sym, start=start, end=end)[field]
            except Exception as e:
                log.warning("Skipping %s: %s", sym, e)
        return pd.DataFrame(series)

    def latest(self, symbol: str) -> pd.Series:
        """Return the most recent bar as a Series."""
        df = self.get(symbol)
        return df.iloc[-1]

    def latest_close(self, symbol: str) -> float:
        """Return the latest closing price."""
        return float(self.get(symbol)["close"].iloc[-1])

    # ── Lookup utilities ──────────────────────────────────────────────────────

    def name(self, ticker: str) -> str:
        """
        Ticker → Chinese stock name.

        >>> api.name("600519")
        '贵州茅台'
        """
        ticker = str(ticker).zfill(6)
        return self._t2n.get(ticker, f"UNKNOWN({ticker})")

    def ticker(self, name: str) -> str:
        """
        Chinese stock name → ticker (exact or partial match).

        >>> api.ticker("贵州茅台")
        '600519'
        >>> api.ticker("茅台")   # partial also works
        '600519'
        """
        # Exact match first
        result = self._n2t.get(name.lower())
        if result:
            return result

        # Partial match
        matches = [(n, t) for n, t in self._n2t.items() if name.lower() in n]
        if len(matches) == 1:
            return matches[0][1]
        if len(matches) > 1:
            names = ", ".join(f"{t}({n})" for n, t in matches[:5])
            raise ValueError(
                f"'{name}' matches multiple stocks: {names}. "
                "Use a more specific name or the ticker directly."
            )
        raise ValueError(f"No stock found matching '{name}'.")

    def search(self, query: str, max_results: int = 20) -> pd.DataFrame:
        """
        Search by partial name or ticker code.

        Returns
        -------
        pd.DataFrame  columns: ticker, name, has_data
        """
        q = query.lower()
        rows = []
        for t, n in self._t2n.items():
            if q in t or q in n.lower():
                rows.append({
                    "ticker":   t,
                    "name":     n,
                    "exchange": _exchange(t),
                    "has_data": (self.root / f"{t}.csv").exists(),
                })
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.sort_values("ticker").head(max_results).reset_index(drop=True)

    def info(self, symbol: str) -> dict:
        """
        Return metadata for a ticker: name, exchange, date range, row count.

        >>> api.info("600519")
        {'ticker': '600519', 'name': '贵州茅台', 'exchange': 'SH',
         'start': '2015-01-05', 'end': '2024-12-31', 'rows': 2436}
        """
        ticker = self._resolve(symbol)
        path   = self.root / f"{ticker}.csv"
        result = {
            "ticker":   ticker,
            "name":     self.name(ticker),
            "exchange": _exchange(ticker),
            "has_data": path.exists(),
        }
        if path.exists():
            df = pd.read_csv(path, parse_dates=["date"], usecols=["date"])
            result["start"] = str(df["date"].min().date())
            result["end"]   = str(df["date"].max().date())
            result["rows"]  = len(df)
        return result

    # ── Universe helpers ──────────────────────────────────────────────────────

    def list_tickers(self, exchange: Optional[str] = None) -> list[str]:
        """
        List all tickers that have a local CSV file.

        Parameters
        ----------
        exchange : "SH" | "SZ" | "BJ" | None (all)
        """
        tickers = sorted(
            p.stem for p in self.root.glob("*.csv")
            if not p.stem.startswith("_") and len(p.stem) == 6 and p.stem.isdigit()
        )
        if exchange:
            tickers = [t for t in tickers if _exchange(t) == exchange.upper()]
        return tickers

    def list_all(self, exchange: Optional[str] = None) -> pd.DataFrame:
        """
        Return a DataFrame of all locally available tickers with names.

        Columns: ticker, name, exchange
        """
        tickers = self.list_tickers(exchange=exchange)
        rows = [{"ticker": t, "name": self.name(t), "exchange": _exchange(t)}
                for t in tickers]
        return pd.DataFrame(rows)

    def universe(self, exchange: Optional[str] = None) -> list[str]:
        """
        Return all tickers known to the name registry, regardless of
        whether a local CSV exists yet.  This is the right source for
        download_all.py — it uses _names.csv (or the embedded dict) so
        no remote API call is needed.

        Parameters
        ----------
        exchange : "SH" | "SZ" | "BJ" | None (all)

        Notes
        -----
        On a brand-new install with no _names.csv, this returns the ~35
        tickers in the embedded _COMMON_NAMES dict.  Run LocalDataAPI
        once with an internet connection to populate _names.csv with the
        full ~5000-stock universe, then subsequent calls are fully offline.
        """
        tickers = sorted(self._t2n.keys())
        if exchange:
            tickers = [t for t in tickers if _exchange(t) == exchange.upper()]
        return tickers

    def available(self, symbol: str) -> bool:
        """Return True if a CSV exists for this ticker."""
        try:
            ticker = self._resolve(symbol)
            return (self.root / f"{ticker}.csv").exists()
        except ValueError:
            return False

    # ── Private ───────────────────────────────────────────────────────────────

    def _resolve(self, symbol: str) -> str:
        """
        Accept ticker OR name, always return a 6-digit ticker string.
        """
        symbol = symbol.strip()
        # Already looks like a ticker
        if symbol.isdigit():
            return symbol.zfill(6)
        # Try name lookup
        return self.ticker(symbol)

    # ── Live / intraday data ─────────────────────────────────────────────────

    def live_bar(
        self,
        symbol: str,
        cutoff_time: str = "14:00",
    ) -> pd.Series:
        """
        Fetch a pseudo-OHLCV bar for today (day t) using intraday minute data
        up to ``cutoff_time``.  Suitable for generating signals before market
        close without using forward-looking close prices.

        The returned Series has the same shape as a row from ``get()``:
            open, high, low, close (= last price at cutoff), volume, as_of

        Parameters
        ----------
        symbol : str
            Ticker or stock name.
        cutoff_time : str
            HH:MM cutoff in China Standard Time (CST).  Default "14:00".
            Bars after this time are excluded.

        Returns
        -------
        pd.Series  with index: open, high, low, close, volume, as_of

        Raises
        ------
        ImportError   if akshare is not installed.
        ValueError    if no intraday data is available (market closed / holiday).

        Notes
        -----
        - Data comes from East Money via ``ak.stock_intraday_em()``.
        - Delay is typically under 1 minute.
        - Only works during Chinese market hours (09:30–15:00 CST).
        - From the UK, you may need a VPN if East Money is blocked.
        """
        try:
            import akshare as ak
        except ImportError:
            raise ImportError("pip install akshare")

        ticker = self._resolve(symbol)

        df = ak.stock_intraday_em(symbol=ticker)
        if df is None or df.empty:
            raise ValueError(
                f"No intraday data for {ticker}. "
                "Market may be closed or the ticker is suspended."
            )

        # Normalise columns (East Money returns Chinese names)
        df = df.rename(columns={
            "时间": "time", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low",  "成交量": "volume",
            "最新价": "close",   # alternate column name
        })

        # Parse time and apply cutoff
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], format="%H:%M:%S", errors="coerce")
            cutoff = pd.to_datetime(cutoff_time, format="%H:%M")
            df = df[df["time"].dt.time <= cutoff.time()]

        if df.empty:
            raise ValueError(
                f"No bars before cutoff {cutoff_time} for {ticker}. "
                "Market may not have opened yet."
            )

        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        bar = pd.Series({
            "open":   float(df["open"].iloc[0])   if "open"   in df.columns else float("nan"),
            "high":   float(df["high"].max())      if "high"   in df.columns else float("nan"),
            "low":    float(df["low"].min())       if "low"    in df.columns else float("nan"),
            "close":  float(df["close"].iloc[-1])  if "close"  in df.columns else float("nan"),
            "volume": float(df["volume"].sum())    if "volume" in df.columns else float("nan"),
            "as_of":  str(df["time"].iloc[-1].time()) if "time" in df.columns else cutoff_time,
        })
        return bar

    def snapshot(self, symbols: Optional[list[str]] = None) -> pd.DataFrame:
        """
        Fetch a real-time quote snapshot for one or more stocks.
        Uses ``ak.stock_zh_a_spot_em()`` which scrapes East Money.

        Parameters
        ----------
        symbols : list of str, optional
            Tickers or names to filter.  If None, returns all available
            (note: East Money currently caps at ~2900 stocks per call).

        Returns
        -------
        pd.DataFrame  columns: ticker, name, price, open, high, low,
                                volume, pct_change, as_of
        """
        try:
            import akshare as ak
        except ImportError:
            raise ImportError("pip install akshare")

        df = ak.stock_zh_a_spot_em()
        if df is None or df.empty:
            raise ValueError("No snapshot data returned from East Money.")

        df = df.rename(columns={
            "代码":   "ticker",
            "名称":   "name",
            "最新价": "price",
            "今开":   "open",
            "最高":   "high",
            "最低":   "low",
            "成交量": "volume",
            "涨跌幅": "pct_change",
        })
        df["ticker"] = df["ticker"].astype(str).str.zfill(6)
        df["as_of"]  = dt.datetime.now().strftime("%H:%M:%S")

        keep = [c for c in ("ticker","name","price","open","high","low",
                            "volume","pct_change","as_of") if c in df.columns]
        df = df[keep]

        if symbols:
            tickers = [self._resolve(s) for s in symbols]
            df = df[df["ticker"].isin(tickers)]

        return df.reset_index(drop=True)

    def __repr__(self) -> str:
        n = len(self.list_tickers())
        return f"LocalDataAPI(root={self.root!r}, files={n})"


# ── Module-level helpers ──────────────────────────────────────────────────────

def _exchange(ticker: str) -> str:
    """Infer exchange from ticker prefix."""
    if ticker.startswith(("60", "68", "90")):
        return "SH"
    elif ticker.startswith(("00", "30", "002")):
        return "SZ"
    elif ticker.startswith(("43", "83", "87")):
        return "BJ"
    return "?"


def returns(df: pd.DataFrame, col: str = "close") -> pd.Series:
    """Daily percentage returns from a price DataFrame."""
    return df[col].pct_change().dropna()


def log_returns(df: pd.DataFrame, col: str = "close") -> pd.Series:
    """Daily log returns."""
    import numpy as np
    return np.log(df[col] / df[col].shift(1)).dropna()


def rolling_vols(
    df: pd.DataFrame,
    windows: list[int] = [5, 10, 20],
    col: str = "close",
    annualise: bool = True,
) -> pd.DataFrame:
    """
    Rolling annualised volatility for multiple windows.

    Returns DataFrame with one column per window, e.g. vol_5, vol_20.
    """
    import numpy as np
    r = returns(df, col)
    factor = np.sqrt(252) if annualise else 1.0
    result = pd.DataFrame(index=df.index)
    for w in windows:
        result[f"vol_{w}"] = r.rolling(w).std() * factor
    return result
