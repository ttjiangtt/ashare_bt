"""
ashare_bt.strategy.base
~~~~~~~~~~~~~~~~~~~~~~~
Abstract base class for all strategies.

To write a custom strategy, subclass ``Strategy`` and implement ``next()``.
Optionally override ``init()`` for one-time setup (precomputing indicators etc.).

Execution model
---------------
- ``buy()``  / ``sell()`` are *market orders* executed at the **next bar's open**.
- ``buy_limit()`` / ``sell_limit()`` are limit orders checked against the next
  bar's high/low.
- All A-share rules are enforced automatically:
    * T+1  — shares bought today cannot be sold until the next trading day.
    * Lot size — orders rounded down to the nearest 100-share lot.
    * Price limits — orders outside ±10 % daily limit are rejected
      (or ±5 % for ST stocks if ``st=True`` on the feed).

Accessing data inside ``next()``
---------------------------------
``self.data`` is the :class:`~ashare_bt.data.feed.DataFeed` with its cursor
pointing at the current bar.  Use ``self.data.close``, ``self.data.history(n)``
etc.

``self.position`` is the current open :class:`Position` (or ``None``).

Example
-------
::

    class GoldenCross(Strategy):
        fast = 5
        slow = 20

        def init(self):
            self.fast_ma = self.indicator(sma, self.data.closes(), self.fast)
            self.slow_ma = self.indicator(sma, self.data.closes(), self.slow)

        def next(self):
            if crossover(self.fast_ma, self.slow_ma):
                self.buy()
            elif crossunder(self.fast_ma, self.slow_ma):
                self.sell()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from broker import Broker
    from position import Position
    from feed import DataFeed


class Strategy(ABC):
    """
    Base class for all strategies.

    Do **not** call ``__init__`` yourself — the engine instantiates
    strategies and injects dependencies automatically.
    """

    def _inject(self, broker: "Broker", data: "DataFeed") -> None:
        self._broker = broker
        self.data = data
        self._indicators: list[np.ndarray] = []

    # ── lifecycle hooks ─────────────────────────────────────────────────────────

    def init(self) -> None:
        """
        Called once before the backtest loop starts.
        Pre-compute indicators here using :meth:`indicator`.
        """

    @abstractmethod
    def next(self) -> None:
        """
        Called on each bar.  Implement your trading logic here.
        """

    # ── indicator helper ────────────────────────────────────────────────────────

    def indicator(self, func, *args, **kwargs) -> np.ndarray:
        """
        Compute a full-series indicator array and register it.

        Parameters
        ----------
        func : callable
            An indicator function from :mod:`ashare_bt.utils.indicators`
            or any callable ``f(*args, **kwargs) -> np.ndarray``.

        Returns
        -------
        np.ndarray
            Full-length array (same length as ``self.data``).

        Example
        -------
        ::

            self.ma5 = self.indicator(sma, self.data.closes(), 5)
        """
        result = func(*args, **kwargs)
        arr = np.asarray(result, dtype=float)
        self._indicators.append(arr)
        return arr

    # ── order helpers ───────────────────────────────────────────────────────────

    def buy(
        self,
        size: Optional[float] = None,
        pct: float = 1.0,
        comment: str = "",
    ) -> None:
        """
        Place a market buy order.

        Parameters
        ----------
        size : int, optional
            Number of shares (rounded to 100-lot).  If None, uses *pct*.
        pct : float
            Fraction of available cash to deploy (0 < pct ≤ 1).
        comment : str
            Tag attached to the resulting trade record.
        """
        self._broker.order(
            direction=1, size=size, pct=pct,
            order_type="market", comment=comment
        )

    def sell(
        self,
        size: Optional[float] = None,
        pct: float = 1.0,
        comment: str = "",
    ) -> None:
        """
        Close (or reduce) the current long position.

        Parameters
        ----------
        size : int, optional
            Shares to sell.  If None, closes the entire position.
        pct : float
            Fraction of the position to close (0 < pct ≤ 1).
        comment : str
            Tag attached to the trade record.
        """
        self._broker.order(
            direction=-1, size=size, pct=pct,
            order_type="market", comment=comment
        )

    def buy_limit(
        self,
        price: float,
        size: Optional[float] = None,
        pct: float = 1.0,
        comment: str = "",
    ) -> None:
        """Limit buy order."""
        self._broker.order(
            direction=1, size=size, pct=pct,
            order_type="limit", limit_price=price, comment=comment
        )

    def sell_limit(
        self,
        price: float,
        size: Optional[float] = None,
        pct: float = 1.0,
        comment: str = "",
    ) -> None:
        """Limit sell order."""
        self._broker.order(
            direction=-1, size=size, pct=pct,
            order_type="limit", limit_price=price, comment=comment
        )

    # ── read-only state ─────────────────────────────────────────────────────────

    @property
    def position(self) -> Optional["Position"]:
        """Current open position, or ``None``."""
        return self._broker.position

    @property
    def cash(self) -> float:
        """Available cash."""
        return self._broker.cash

    @property
    def equity(self) -> float:
        """Total portfolio value (cash + position market value)."""
        return self._broker.equity

    # ── signal utilities ────────────────────────────────────────────────────────

    @staticmethod
    def crossover(a: np.ndarray, b: np.ndarray) -> bool:
        """True if series *a* crossed **above** series *b* on the last bar."""
        return (
            len(a) >= 2
            and not np.isnan(a[-2]) and not np.isnan(b[-2])
            and a[-2] < b[-2] and a[-1] >= b[-1]
        )

    @staticmethod
    def crossunder(a: np.ndarray, b: np.ndarray) -> bool:
        """True if series *a* crossed **below** series *b* on the last bar."""
        return (
            len(a) >= 2
            and not np.isnan(a[-2]) and not np.isnan(b[-2])
            and a[-2] > b[-2] and a[-1] <= b[-1]
        )

    def __repr__(self) -> str:
        params = {k: v for k, v in self.__class__.__dict__.items()
                  if not k.startswith("_") and not callable(v)}
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{self.__class__.__name__}({param_str})"


# ── module-level convenience wrappers ──────────────────────────────────────────

def crossover(a: np.ndarray, b) -> bool:
    """Standalone crossover check (works with scalar *b*)."""
    b_arr = np.full_like(a, b) if np.isscalar(b) else b
    return Strategy.crossover(a, b_arr)


def crossunder(a: np.ndarray, b) -> bool:
    """Standalone crossunder check (works with scalar *b*)."""
    b_arr = np.full_like(a, b) if np.isscalar(b) else b
    return Strategy.crossunder(a, b_arr)
