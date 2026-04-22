"""
ashare_bt.data.loader
~~~~~~~~~~~~~~~~~~~~~
Downloads A-share OHLCV data via AKShare and caches it locally.

Features
--------
* No API key required — AKShare wraps 东方财富 / 新浪财经 for free.
* Local disk cache (CSV or Parquet) so repeated runs are instant.
* Incremental updates — only fetches the missing tail of the series.
* Normalises columns to the DataFeed schema automatically.
* Helper to search the full A-share stock list by name or code.
* Index data (上证、深证、沪深300 etc.)

Quick start
-----------
::

    from ashare_bt.data.loader import AKLoader

    loader = AKLoader(cache_dir="./data_cache")

    # Single stock
    feed = loader.load("600519", start="2020-01-01", adjust="qfq")

    # Multiple stocks
    feeds = loader.load_batch(["000001", "600519", "601318"],
                               start="2020-01-01", adjust="qfq")

    # Index
    feed_index = loader.load_index("000300", start="2020-01-01")  # CSI 300

    # Search
    results = loader.search("茅台")
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import pandas as pd

from feed import DataFeed

logger = logging.getLogger(__name__)

AdjustType = Literal["", "qfq", "hfq"]   # none / 前复权 / 后复权

# ── Column rename map: AKShare → DataFeed canonical ────────────────────────────
_AK_RENAME = {
    "日期":   "date",
    "开盘":   "open",
    "收盘":   "close",
    "最高":   "high",
    "最低":   "low",
    "成交量": "volume",
    "成交额": "amount",
    "换手率": "turnover",
    "涨跌幅": "pct_change",
    "股票代码": "symbol_col",  # drop later
}

_INDEX_RENAME = {
    "日期":   "date",
    "开盘":   "open",
    "收盘":   "close",
    "最高":   "high",
    "最低":   "low",
    "成交量": "volume",
}


def _normalise_ak(df: pd.DataFrame) -> pd.DataFrame:
    """Rename AKShare columns to canonical names and clean types."""
    df = df.rename(columns=_AK_RENAME)
    # Drop non-canonical extras
    drop = [c for c in ("symbol_col", "振幅", "涨跌额", "turnover", "pct_change")
            if c in df.columns]
    df = df.drop(columns=drop, errors="ignore")
    df["date"] = pd.to_datetime(df["date"])
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _normalise_ak_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=_INDEX_RENAME)
    df["date"] = pd.to_datetime(df["date"])
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # index data may not have open/high/low — forward fill if missing
    for col in ("open", "high", "low"):
        if col not in df.columns:
            df[col] = df["close"]
    if "volume" not in df.columns:
        df["volume"] = 0
    df = df.sort_values("date").reset_index(drop=True)
    return df


class AKLoader:
    """
    Data loader backed by AKShare with optional local cache.

    Parameters
    ----------
    cache_dir : str | Path | None
        Directory for cached CSV/Parquet files.
        Pass ``None`` to disable caching (always download fresh).
    fmt : str
        Cache file format: ``'csv'`` (default) or ``'parquet'``
        (requires ``pyarrow`` or ``fastparquet``).
    retry : int
        Number of retries on network error.
    retry_delay : float
        Seconds to wait between retries.
    throttle : float
        Seconds to sleep between API calls in ``load_batch()``
        to avoid rate-limiting.
    """

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = "./data_cache",
        fmt: Literal["csv", "parquet"] = "csv",
        retry: int = 3,
        retry_delay: float = 2.0,
        throttle: float = 0.5,
    ) -> None:
        self.fmt = fmt
        self.retry = retry
        self.retry_delay = retry_delay
        self.throttle = throttle

        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

        self._ak = self._import_akshare()

    # ── public API ───────────────────────────────────────────────────────────────

    def load(
        self,
        symbol: str,
        start: str = "2015-01-01",
        end: Optional[str] = None,
        adjust: AdjustType = "qfq",
        force_refresh: bool = False,
    ) -> DataFeed:
        """
        Load daily OHLCV data for a single A-share stock.

        Parameters
        ----------
        symbol : str
            6-digit A-share code, e.g. ``"600519"`` or ``"000001"``.
        start : str
            Start date ``"YYYY-MM-DD"``.
        end : str, optional
            End date ``"YYYY-MM-DD"``.  Defaults to today.
        adjust : str
            Adjustment type: ``""`` (unadjusted), ``"qfq"`` (前复权),
            ``"hfq"`` (后复权).  Default is ``"qfq"``.
        force_refresh : bool
            Ignore cache and re-download from scratch.

        Returns
        -------
        DataFeed
        """
        symbol = symbol.strip().zfill(6)
        end = end or date.today().strftime("%Y-%m-%d")

        df = self._load_ohlcv(symbol, start, end, adjust, force_refresh)
        # Trim to requested range
        df = df[(df["date"] >= pd.Timestamp(start)) &
                (df["date"] <= pd.Timestamp(end))].reset_index(drop=True)

        if df.empty:
            raise ValueError(
                f"No data returned for {symbol} "
                f"({start} → {end}, adjust={adjust!r}). "
                "Check the symbol code and date range."
            )

        logger.info("Loaded %s: %d bars (%s → %s)",
                    symbol, len(df), df['date'].iloc[0].date(), df['date'].iloc[-1].date())
        return DataFeed(df, symbol=symbol)

    def load_batch(
        self,
        symbols: List[str],
        start: str = "2015-01-01",
        end: Optional[str] = None,
        adjust: AdjustType = "qfq",
        force_refresh: bool = False,
    ) -> Dict[str, DataFeed]:
        """
        Load multiple stocks, returning a dict keyed by symbol.

        A throttle delay is applied between calls to avoid rate-limiting.
        Failed symbols are skipped with a warning rather than raising.

        Returns
        -------
        dict[str, DataFeed]
        """
        result: Dict[str, DataFeed] = {}
        for i, sym in enumerate(symbols):
            if i > 0:
                time.sleep(self.throttle)
            try:
                result[sym] = self.load(sym, start=start, end=end,
                                         adjust=adjust, force_refresh=force_refresh)
            except Exception as exc:
                warnings.warn(f"Failed to load {sym}: {exc}")
        return result

    def load_index(
        self,
        symbol: str = "000300",
        start: str = "2015-01-01",
        end: Optional[str] = None,
        force_refresh: bool = False,
    ) -> DataFeed:
        """
        Load a Chinese stock index.

        Common symbols
        --------------
        ======== =================
        000001   上证综指
        399001   深证成指
        000300   沪深300
        000905   中证500
        000852   中证1000
        399006   创业板指
        ======== =================

        Parameters
        ----------
        symbol : str
            Index code (6 digits, no exchange prefix).
        start : str
        end : str, optional
        force_refresh : bool

        Returns
        -------
        DataFeed
        """
        end = end or date.today().strftime("%Y-%m-%d")
        cache_key = f"index_{symbol}"
        df = self._load_with_cache(
            cache_key=cache_key,
            fetch_fn=lambda s, e: self._fetch_index(symbol, s, e),
            start=start,
            end=end,
            force_refresh=force_refresh,
        )
        df = df[(df["date"] >= pd.Timestamp(start)) &
                (df["date"] <= pd.Timestamp(end))].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No index data for {symbol} in range {start} → {end}.")
        return DataFeed(df, symbol=f"IDX_{symbol}")

    def search(self, query: str, max_results: int = 20) -> pd.DataFrame:
        """
        Search the full A-share stock list by name or code.

        Parameters
        ----------
        query : str
            Partial name (e.g. ``"茅台"``) or code (e.g. ``"6005"``).

        Returns
        -------
        pd.DataFrame  with columns: code, name, exchange
        """
        stock_list = self._get_stock_list()
        mask = (
            stock_list["code"].str.contains(query, case=False, na=False)
            | stock_list["name"].str.contains(query, case=False, na=False)
        )
        return stock_list[mask].head(max_results).reset_index(drop=True)

    def stock_list(self) -> pd.DataFrame:
        """Return the full A-share universe (code, name, exchange)."""
        return self._get_stock_list()

    # ── private helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _import_akshare():
        try:
            import akshare as ak
            return ak
        except ImportError:
            raise ImportError(
                "AKShare is required for data download.\n"
                "Install it with:  pip install akshare\n"
                "or add it to environment.yml and run: conda env update -f environment.yml"
            )

    def _cache_path(self, key: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        ext = "parquet" if self.fmt == "parquet" else "csv"
        return self.cache_dir / f"{key}.{ext}"

    def _read_cache(self, path: Path) -> Optional[pd.DataFrame]:
        if not path.exists():
            return None
        try:
            if path.suffix == ".parquet":
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path, parse_dates=["date"])
            return df
        except Exception as exc:
            warnings.warn(f"Cache read failed for {path}: {exc} — will re-download.")
            return None

    def _write_cache(self, path: Path, df: pd.DataFrame) -> None:
        try:
            if path.suffix == ".parquet":
                df.to_parquet(path, index=False)
            else:
                df.to_csv(path, index=False)
        except Exception as exc:
            warnings.warn(f"Cache write failed for {path}: {exc}")

    def _load_ohlcv(
        self,
        symbol: str,
        start: str,
        end: str,
        adjust: str,
        force_refresh: bool,
    ) -> pd.DataFrame:
        cache_key = f"{symbol}_{adjust or 'raw'}"
        return self._load_with_cache(
            cache_key=cache_key,
            fetch_fn=lambda s, e: self._fetch_stock(symbol, s, e, adjust),
            start=start,
            end=end,
            force_refresh=force_refresh,
        )

    def _load_with_cache(
        self,
        cache_key: str,
        fetch_fn,
        start: str,
        end: str,
        force_refresh: bool,
    ) -> pd.DataFrame:
        path = self._cache_path(cache_key)

        if not force_refresh and path is not None:
            cached = self._read_cache(path)
            if cached is not None and not cached.empty:
                last_cached = cached["date"].max()
                end_ts = pd.Timestamp(end)
                today = pd.Timestamp(date.today())

                # If we need more recent data, fetch only the missing tail
                if last_cached < min(end_ts, today - timedelta(days=1)):
                    fetch_start = (last_cached + timedelta(days=1)).strftime("%Y-%m-%d")
                    logger.info("Incremental update for %s from %s", cache_key, fetch_start)
                    new_data = self._retry_fetch(fetch_fn, fetch_start, end)
                    if new_data is not None and not new_data.empty:
                        combined = pd.concat([cached, new_data], ignore_index=True)
                        combined = combined.drop_duplicates("date").sort_values("date")
                        if path is not None:
                            self._write_cache(path, combined)
                        return combined
                return cached

        # Full download
        logger.info("Downloading %s (%s → %s)", cache_key, start, end)
        df = self._retry_fetch(fetch_fn, start, end)
        if df is None or df.empty:
            return pd.DataFrame()
        if path is not None:
            self._write_cache(path, df)
        return df

    def _retry_fetch(self, fetch_fn, start: str, end: str) -> Optional[pd.DataFrame]:
        last_exc = None
        for attempt in range(1, self.retry + 1):
            try:
                return fetch_fn(start, end)
            except Exception as exc:
                last_exc = exc
                if attempt < self.retry:
                    logger.warning("Attempt %d failed: %s — retrying in %.1fs",
                                   attempt, exc, self.retry_delay)
                    time.sleep(self.retry_delay)
        raise RuntimeError(f"All {self.retry} download attempts failed.") from last_exc

    def _fetch_stock(
        self, symbol: str, start: str, end: str, adjust: str
    ) -> pd.DataFrame:
        ak = self._ak
        start_fmt = start.replace("-", "")
        end_fmt   = end.replace("-", "")

        # Primary: 东方财富 (East Money)
        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_fmt,
                end_date=end_fmt,
                adjust=adjust,
            )
            return _normalise_ak(df)
        except Exception as e:
            logger.warning("East Money API failed (%s), trying Sina fallback…", e)

        # Fallback: Sina Finance — more accessible outside China
        exchange = "sh" if symbol.startswith(("6", "9")) else "sz"
        sina_symbol = f"{exchange}{symbol}"
        adjust_sina = adjust if adjust else None
        df = ak.stock_zh_a_daily(
            symbol=sina_symbol,
            start_date=start_fmt,
            end_date=end_fmt,
            adjust=adjust_sina,
        )
        df = df.rename(columns={"date": "date", "open": "open", "high": "high",
                                 "low": "low", "close": "close", "volume": "volume"})
        df["date"] = pd.to_datetime(df["date"])
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.sort_values("date").reset_index(drop=True)

    def _fetch_index(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        ak = self._ak
        start_fmt = start.replace("-", "")
        end_fmt   = end.replace("-", "")
        try:
            # East Money index API (covers most indices)
            df = ak.index_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_fmt,
                end_date=end_fmt,
            )
        except Exception:
            # Fallback: Sina index API
            exchange = "sh" if symbol.startswith(("000", "0008", "0009")) else "sz"
            df = ak.stock_zh_index_daily(symbol=f"{exchange}{symbol}")
        return _normalise_ak_index(df)

    def _get_stock_list(self) -> pd.DataFrame:
        """Fetch the full A-share universe from AKShare (cached in memory)."""
        if hasattr(self, "_stock_list_cache"):
            return self._stock_list_cache

        ak = self._ak
        try:
            df = ak.stock_zh_a_spot_em()
            result = pd.DataFrame({
                "code": df["代码"].astype(str).str.zfill(6),
                "name": df["名称"],
                "price": df["最新价"],
            })
        except Exception:
            # Fallback
            df = ak.stock_info_a_code_name()
            result = pd.DataFrame({
                "code": df["code"].astype(str).str.zfill(6),
                "name": df["name"],
            })

        # Infer exchange from code prefix
        def _exchange(code: str) -> str:
            if code.startswith(("60", "68")):
                return "SH"
            elif code.startswith(("00", "30", "002")):
                return "SZ"
            elif code.startswith("8"):
                return "BJ"
            return "?"

        result["exchange"] = result["code"].apply(_exchange)
        self._stock_list_cache = result
        return result
