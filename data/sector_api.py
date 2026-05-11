"""
data/sector_api.py
~~~~~~~~~~~~~~~~~~
A-share sector classification API with local caching.

Supports three classification schemes:
  sw   — 申万行业 (Shenwan) — industry standard, 3 levels (L1/L2/L3)
  em   — 东方财富行业板块 (East Money) — board-level classification
  csrc — 证监会行业分类 (CSRC) — regulatory classification

All data is cached locally as CSV files and only re-fetched when stale
or when force_refresh=True.

Quick start
-----------
    from data.sector_api import SectorAPI

    api = SectorAPI(cache_dir="./sector_cache")

    # Ticker → sector
    api.sector("600519")           # → "食品饮料" (SW L1)
    api.sector("600519", scheme="em")   # → "白酒"
    api.sector("600519", level=2)       # → "白酒" (SW L2)

    # Sector → tickers
    api.tickers("食品饮料")        # → ["600519", "000858", ...]
    api.tickers("白酒", level=2)

    # Add sector column to a DataFrame
    df = api.enrich(df)            # adds sw_l1, sw_l2, em_sector columns

    # Full classification table
    api.sw_table()                 # DataFrame: ticker, name, sw_l1, sw_l2, sw_l3
    api.em_table()                 # DataFrame: ticker, name, em_sector

    # List all sector names
    api.list_sectors()             # SW L1 sectors
    api.list_sectors(level=2)      # SW L2 sectors

    # Refresh cache
    api.refresh()
"""

from __future__ import annotations

import logging
import warnings
from datetime import date, timedelta
from pathlib import Path
from typing import List, Literal, Optional, Union

import pandas as pd

log = logging.getLogger(__name__)

SchemeType = Literal["sw", "em", "csrc"]


# ── SectorAPI ─────────────────────────────────────────────────────────────────

class SectorAPI:
    """
    A-share sector classification with local disk cache.

    Parameters
    ----------
    cache_dir : str | Path
        Directory for cached CSV files.
        Default: ``./sector_cache`` (created automatically).
    stale_days : int
        Re-fetch from remote if cache is older than this many days.
        Default 7 — sector classifications rarely change.
    """

    def __init__(
        self,
        cache_dir: Union[str, Path] = "./sector_cache",
        stale_days: int = 7,
    ) -> None:
        self.cache_dir  = Path(cache_dir)
        self.stale_days = stale_days
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory caches (populated on first access)
        self._sw:   Optional[pd.DataFrame] = None   # full SW table
        self._em:   Optional[pd.DataFrame] = None   # full EM table
        self._csrc: Optional[pd.DataFrame] = None   # full CSRC table

    # ── Public lookup API ─────────────────────────────────────────────────────

    def sector(
        self,
        ticker: str,
        scheme: SchemeType = "sw",
        level: int = 1,
    ) -> str:
        """
        Return the sector name for a ticker.

        Parameters
        ----------
        ticker : str   6-digit code.
        scheme : str   ``'sw'``, ``'em'``, or ``'csrc'``.
        level  : int   1, 2, or 3 (SW only; ignored for em/csrc).

        Returns
        -------
        str  sector name, or ``'Unknown'`` if not found.

        Example
        -------
        >>> api.sector("600519")          # → '食品饮料'
        >>> api.sector("600519", level=2) # → '白酒'
        """
        ticker = str(ticker).zfill(6)
        if scheme == "sw":
            df  = self.sw_table()
            col = f"sw_l{level}"
            row = df[df["ticker"] == ticker]
            return str(row[col].iloc[0]) if not row.empty else "Unknown"
        elif scheme == "em":
            df  = self.em_table()
            row = df[df["ticker"] == ticker]
            return str(row["em_sector"].iloc[0]) if not row.empty else "Unknown"
        elif scheme == "csrc":
            df  = self.csrc_table()
            row = df[df["ticker"] == ticker]
            return str(row["csrc_sector"].iloc[0]) if not row.empty else "Unknown"
        else:
            raise ValueError(f"scheme must be 'sw', 'em', or 'csrc', got {scheme!r}")

    def tickers(
        self,
        sector_name: str,
        scheme: SchemeType = "sw",
        level: int = 1,
    ) -> List[str]:
        """
        Return all tickers in a sector.

        Parameters
        ----------
        sector_name : str   Partial or full sector name (case-insensitive).
        scheme : str
        level  : int        SW level to match against (1, 2, or 3).

        Returns
        -------
        list of str  6-digit ticker codes.

        Example
        -------
        >>> api.tickers("食品饮料")
        >>> api.tickers("白酒", level=2)
        """
        q = sector_name.lower()
        if scheme == "sw":
            df  = self.sw_table()
            col = f"sw_l{level}"
            mask = df[col].str.lower().str.contains(q, na=False)
            return df[mask]["ticker"].tolist()
        elif scheme == "em":
            df   = self.em_table()
            mask = df["em_sector"].str.lower().str.contains(q, na=False)
            return df[mask]["ticker"].tolist()
        elif scheme == "csrc":
            df   = self.csrc_table()
            mask = df["csrc_sector"].str.lower().str.contains(q, na=False)
            return df[mask]["ticker"].tolist()

    def list_sectors(
        self,
        scheme: SchemeType = "sw",
        level: int = 1,
    ) -> List[str]:
        """
        List all unique sector names.

        Example
        -------
        >>> api.list_sectors()           # SW L1 sectors
        >>> api.list_sectors(level=2)    # SW L2 sectors
        >>> api.list_sectors("em")
        """
        if scheme == "sw":
            df  = self.sw_table()
            col = f"sw_l{level}"
            return sorted(df[col].dropna().unique().tolist())
        elif scheme == "em":
            df = self.em_table()
            return sorted(df["em_sector"].dropna().unique().tolist())
        elif scheme == "csrc":
            df = self.csrc_table()
            return sorted(df["csrc_sector"].dropna().unique().tolist())

    def enrich(
        self,
        df: pd.DataFrame,
        ticker_col: str = "ticker",
        schemes: List[str] = ("sw", "em"),
        sw_levels: List[int] = (1, 2),
    ) -> pd.DataFrame:
        """
        Add sector columns to a DataFrame that has a ticker column.

        Parameters
        ----------
        df          : DataFrame with a ticker column.
        ticker_col  : name of the ticker column (default ``'ticker'``).
        schemes     : which schemes to add.
        sw_levels   : which SW levels to add.

        Returns
        -------
        DataFrame with added columns: ``sw_l1``, ``sw_l2``, ``em_sector``.

        Example
        -------
        >>> enriched = api.enrich(signals_df)
        """
        result = df.copy()
        tickers = result[ticker_col].astype(str).str.zfill(6)

        if "sw" in schemes:
            sw = self.sw_table().set_index("ticker")
            for lvl in sw_levels:
                col = f"sw_l{lvl}"
                if col in sw.columns:
                    result[col] = tickers.map(sw[col]).fillna("Unknown")

        if "em" in schemes:
            em = self.em_table().set_index("ticker")
            result["em_sector"] = tickers.map(em["em_sector"]).fillna("Unknown")

        if "csrc" in schemes:
            csrc = self.csrc_table().set_index("ticker")
            result["csrc_sector"] = tickers.map(csrc["csrc_sector"]).fillna("Unknown")

        return result

    def search(self, query: str) -> pd.DataFrame:
        """
        Search sector classifications by ticker or sector name.

        Returns a merged view across all schemes.
        """
        q = query.lower()
        sw   = self.sw_table()
        em   = self.em_table()
        csrc = self.csrc_table()

        # Merge all schemes on ticker
        merged = sw.merge(em[["ticker","em_sector"]], on="ticker", how="left")
        merged = merged.merge(csrc[["ticker","csrc_sector"]], on="ticker", how="left")

        # Filter by ticker or any sector column
        mask = merged["ticker"].str.contains(q, na=False)
        for col in ["name","sw_l1","sw_l2","sw_l3","em_sector","csrc_sector"]:
            if col in merged.columns:
                mask = mask | merged[col].str.lower().str.contains(q, na=False)

        return merged[mask].reset_index(drop=True)

    # ── Table accessors ───────────────────────────────────────────────────────

    def sw_table(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Full Shenwan classification table.

        Returns
        -------
        DataFrame  columns: ticker, name, sw_l1, sw_l2, sw_l3
        """
        if self._sw is None or force_refresh:
            self._sw = self._load_or_fetch("sw", self._fetch_sw)
        return self._sw

    def em_table(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        East Money industry classification table.

        Returns
        -------
        DataFrame  columns: ticker, name, em_sector
        """
        if self._em is None or force_refresh:
            try:
                self._em = self._load_or_fetch("em", self._fetch_em)
            except RuntimeError as e:
                log.warning("EM table unavailable: %s — returning empty", e)
                self._em = pd.DataFrame(columns=["ticker","name","em_sector"])
        return self._em

    def csrc_table(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        CSRC (证监会) classification table.

        Returns
        -------
        DataFrame  columns: ticker, name, csrc_sector, csrc_code
        """
        if self._csrc is None or force_refresh:
            try:
                self._csrc = self._load_or_fetch("csrc", self._fetch_csrc)
            except RuntimeError as e:
                log.warning("CSRC table unavailable: %s — returning empty", e)
                self._csrc = pd.DataFrame(columns=["ticker","name","csrc_sector","csrc_code"])
        return self._csrc

    def refresh(self) -> None:
        """Force re-fetch all schemes from remote."""
        log.info("Refreshing all sector classifications…")
        self._sw   = self._load_or_fetch("sw",   self._fetch_sw,   force=True)
        self._em   = self._load_or_fetch("em",   self._fetch_em,   force=True)
        self._csrc = self._load_or_fetch("csrc", self._fetch_csrc, force=True)
        log.info("Sector data refreshed.")

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"sector_{key}.csv"

    def _is_stale(self, path: Path) -> bool:
        if not path.exists():
            return True
        mtime = date.fromtimestamp(path.stat().st_mtime)
        return (date.today() - mtime).days >= self.stale_days

    def _load_or_fetch(self, key: str, fetch_fn, force: bool = False) -> pd.DataFrame:
        path = self._cache_path(key)
        if not force and not self._is_stale(path):
            try:
                df = pd.read_csv(path, dtype=str)
                log.debug("Loaded %s from cache (%s)", key, path)
                return df
            except Exception as e:
                log.warning("Cache read failed for %s: %s", key, e)

        log.info("Fetching %s sector data from remote…", key)
        try:
            df = fetch_fn()
            df.to_csv(path, index=False)
            log.info("Saved %s to cache (%d rows)", key, len(df))
            return df
        except Exception as e:
            log.warning("Remote fetch failed for %s: %s — trying cache…", key, e)
            if path.exists():
                return pd.read_csv(path, dtype=str)
            raise RuntimeError(f"No cache and remote fetch failed for {key}: {e}")

    # ── Remote fetch functions ────────────────────────────────────────────────

    def _fetch_sw(self) -> pd.DataFrame:
        """
        Fetch Shenwan classification for all A-shares.

        Primary:  stock_industry_clf_hist_sw() — returns all stocks with
                  SW L1/L2/L3 in one call (no looping needed).
        Fallback: loop over L1 sectors using sw_index_cons().
        """
        try:
            import akshare as ak
        except ImportError:
            raise ImportError("pip install akshare")

        # Try multiple function names — AKShare renames functions across versions.
        # We probe which ones exist and use the first that works.
        attempts = [
            # Option 1: full classification in one call (newest)
            ("stock_industry_clf_hist_sw", None),
            # Option 2: index_component_sw with L1 code loop
            ("index_component_sw", "loop_l1"),
            # Option 3: sw_index_cons with L1 code loop
            ("sw_index_cons", "loop_l1"),
        ]

        # stock_industry_clf_hist_sw returns sector-level summary rows, NOT per-stock.
        # Skip it and go straight to the per-stock constituent loop below.

        # ── Options 2 & 3: loop L1 sectors via index_component_sw or sw_index_cons ──
        loop_fn = None
        for fn_name in ("index_component_sw", "sw_index_cons"):
            if hasattr(ak, fn_name):
                loop_fn = getattr(ak, fn_name)
                log.info("SW: using %s for L1 loop", fn_name)
                break

        if loop_fn is None:
            raise RuntimeError(
                "No SW constituent function found. "
                f"AKShare version: {ak.__version__}. "
                "Try: pip install akshare --upgrade"
            )

        rows = []
        try:
            l1_info = ak.sw_index_first_info()
            l1_codes = l1_info["行业代码"].tolist()
            l1_names = dict(zip(l1_info["行业代码"], l1_info["行业名称"]))
        except Exception as e:
            raise RuntimeError(f"Failed to fetch SW L1 list: {e}")

        for l1_code in l1_codes:
            l1_name   = l1_names.get(l1_code, "")
            code_clean = l1_code.replace(".SI", "")
            cons = None
            for kwargs in ({"symbol": code_clean}, {"index_code": code_clean}):
                try:
                    cons = loop_fn(**kwargs)
                    if cons is not None and not cons.empty:
                        break
                except Exception as e:
                    log.debug("SW loop %s kwargs=%s: %s", l1_code, kwargs, e)
            if cons is None or cons.empty:
                log.warning("SW loop returned no data for %s", l1_code)
                continue

            if cons is None or cons.empty:
                continue

            for col in ("成分股代码","股票代码","code"):
                if col in cons.columns:
                    cons = cons.rename(columns={col: "ticker"})
                    break
            for col in ("成分股名称","股票名称","name"):
                if col in cons.columns:
                    cons = cons.rename(columns={col: "name"})
                    break

            for _, row in cons.iterrows():
                ticker = str(row.get("ticker","")).replace(".SZ","").replace(".SH","")
                ticker = ticker[-6:].zfill(6)
                rows.append({"ticker": ticker, "name": str(row.get("name","")),
                             "sw_l1": l1_name, "sw_l2": "", "sw_l3": ""})

        # Best-effort L2 from East Money spot data
        try:
            spot = ak.stock_zh_a_spot_em()
            if spot is not None and "行业" in spot.columns and "代码" in spot.columns:
                spot["ticker"] = spot["代码"].astype(str).str.zfill(6)
                l2_map = dict(zip(spot["ticker"], spot["行业"]))
                for r in rows:
                    r["sw_l2"] = l2_map.get(r["ticker"], "")
        except Exception:
            pass

        df = pd.DataFrame(rows)
        if df.empty:
            raise RuntimeError("SW fetch returned no data from any source")
        df = df.drop_duplicates("ticker").reset_index(drop=True)
        log.info("SW: %d tickers via loop", len(df))
        return df[["ticker","name","sw_l1","sw_l2","sw_l3"]]

    def _fetch_em(self) -> pd.DataFrame:
        """
        Fetch East Money industry board classification.
        Each ticker → board name (e.g. "白酒", "半导体", "新能源车").
        """
        try:
            import akshare as ak
        except ImportError:
            raise ImportError("pip install akshare")

        rows = []

        # Get list of all EM industry boards
        try:
            boards = ak.stock_board_industry_name_em()
        except Exception as e:
            raise RuntimeError(f"Failed to fetch EM board list: {e}")

        for _, board_row in boards.iterrows():
            board_name = board_row.get("板块名称", board_row.iloc[1])
            try:
                cons = ak.stock_board_industry_cons_em(symbol=board_name)
                if cons is None or cons.empty:
                    continue
                for col in ("代码","股票代码","code"):
                    if col in cons.columns:
                        cons = cons.rename(columns={col: "ticker"})
                        break
                for col in ("名称","股票名称","name"):
                    if col in cons.columns:
                        cons = cons.rename(columns={col: "name"})
                        break
                for _, row in cons.iterrows():
                    ticker = str(row.get("ticker","")).zfill(6)
                    rows.append({
                        "ticker":    ticker,
                        "name":      str(row.get("name","")),
                        "em_sector": board_name,
                    })
            except Exception as e:
                log.debug("EM constituent fetch failed for %s: %s", board_name, e)
                continue

        df = pd.DataFrame(rows)
        if df.empty:
            raise RuntimeError("EM fetch returned no data")
        # Keep last (most specific) entry per ticker if duplicates
        df = df.drop_duplicates("ticker", keep="last").reset_index(drop=True)
        return df[["ticker","name","em_sector"]]

    def _fetch_csrc(self) -> pd.DataFrame:
        """
        Fetch CSRC (证监会) industry classification via 巨潮资讯.
        """
        try:
            import akshare as ak
        except ImportError:
            raise ImportError("pip install akshare")

        # Try with and without the symbol argument — varies by version
        df = None
        for kwargs in ({"symbol": "沪深A股"}, {"symbol": "A股"}, {}):
            try:
                df = ak.stock_industry_category_cninfo(**kwargs)
                if df is not None and not df.empty:
                    break
            except Exception as e:
                log.debug("CSRC kwargs=%s failed: %s", kwargs, e)
        if df is None or df.empty:
            raise RuntimeError("Failed to fetch CSRC classification from any call variant")

        df = df.rename(columns={
            "股票代码":    "ticker",
            "股票简称":    "name",
            "行业分类名称": "csrc_sector",
            "行业分类代码": "csrc_code",
        })
        df["ticker"] = df["ticker"].astype(str).str.zfill(6)
        keep = [c for c in ("ticker","name","csrc_sector","csrc_code") if c in df.columns]
        df = df[keep].drop_duplicates("ticker").reset_index(drop=True)
        return df

    def __repr__(self) -> str:
        cached = [k for k in ("sw","em","csrc")
                  if self._cache_path(k).exists()]
        return f"SectorAPI(cache_dir={self.cache_dir!r}, cached={cached})"
