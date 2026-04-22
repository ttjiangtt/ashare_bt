"""
ashare_bt.engine.broker
~~~~~~~~~~~~~~~~~~~~~~~
Simulated brokerage with A-share specific rules:

* T+1  — shares bought on bar N cannot be sold before bar N+1.
* Lot size — all orders rounded to the nearest 100-share lot (手).
* Daily price limit — ±10 % from previous close (±5 % for ST stocks).
* Commission — applied on both buy and sell (default 0.03 %).
* Stamp duty — sell-side only (default 0.10 %).  Set to 0 to disable.
* Slippage — fractional price impact (default 0.05 %).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, List

import pandas as pd

from position import Position, Trade


_LOT = 100   # shares per lot (手)


@dataclass
class _PendingOrder:
    direction: int          # +1 buy, -1 sell
    size: Optional[int]     # None ⇒ use pct
    pct: float
    order_type: str         # 'market' | 'limit'
    limit_price: Optional[float]
    comment: str


class Broker:
    """
    Single-stock paper broker.

    Parameters
    ----------
    cash : float
    commission_rate : float   Per-side commission rate (fraction, e.g. 0.0003).
    stamp_duty : float        Sell-side only (fraction).
    slippage : float          Fractional price impact per order.
    price_limit : float       Daily limit (0.10 for most A-shares).
    """

    def __init__(
        self,
        cash: float,
        commission_rate: float = 0.0003,
        stamp_duty: float = 0.001,
        slippage: float = 0.0005,
        price_limit: float = 0.10,
        symbol: str = "UNKNOWN",
    ) -> None:
        self._initial_cash = cash
        self._cash = cash
        self._commission_rate = commission_rate
        self._stamp_duty = stamp_duty
        self._slippage = slippage
        self._price_limit = price_limit
        self.symbol = symbol

        self._position: Optional[Position] = None
        self._pending: Optional[_PendingOrder] = None
        self._trades: List[Trade] = []
        self._trade_counter = 0

        # Set by engine each bar
        self._bar_index: int = 0
        self._prev_close: float = 0.0

    # ── public API ──────────────────────────────────────────────────────────────

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def position(self) -> Optional[Position]:
        return self._position

    @property
    def trades(self) -> List[Trade]:
        return self._trades

    def equity(self, price: float) -> float:
        pos_value = self._position.market_value(price) if self._position else 0.0
        return self._cash + pos_value

    # ── order placement ─────────────────────────────────────────────────────────

    def order(
        self,
        direction: int,
        size: Optional[float],
        pct: float,
        order_type: str,
        limit_price: Optional[float] = None,
        comment: str = "",
    ) -> None:
        """Queue an order to be executed at the start of the *next* bar."""
        self._pending = _PendingOrder(
            direction=direction,
            size=int(size) if size is not None else None,
            pct=pct,
            order_type=order_type,
            limit_price=limit_price,
            comment=comment,
        )

    # ── called by engine ────────────────────────────────────────────────────────

    def process_pending(self, bar: pd.Series, bar_index: int) -> None:
        """
        Attempt to fill any queued order using *bar*'s prices.

        Called at the **start** of each bar (after the strategy has run
        on the previous bar).
        """
        if self._pending is None:
            return

        o = self._pending
        self._pending = None
        open_price = float(bar["open"])
        high = float(bar["high"])
        low = float(bar["low"])

        # Price limit check
        if self._prev_close > 0:
            upper = self._prev_close * (1 + self._price_limit)
            lower = self._prev_close * (1 - self._price_limit)
            if open_price > upper * 1.001:
                return  # stuck at limit-up; can't buy
            if open_price < lower * 0.999:
                return  # stuck at limit-down; can't sell

        if o.order_type == "limit":
            if o.direction == 1 and low > o.limit_price:
                return  # price never hit limit
            if o.direction == -1 and high < o.limit_price:
                return
            exec_price = o.limit_price
        else:
            # Market order: fill at open with slippage
            slip = self._slippage * open_price
            exec_price = open_price + slip if o.direction == 1 else open_price - slip

        if o.direction == 1:
            self._execute_buy(exec_price, o.size, o.pct, bar_index, bar.name, o.comment)
        else:
            self._execute_sell(exec_price, bar_index, bar.name, o.comment)

    def mark_eod(self, prev_close: float) -> None:
        self._prev_close = prev_close

    # ── private execution ───────────────────────────────────────────────────────

    def _execute_buy(
        self,
        price: float,
        size: Optional[int],
        pct: float,
        bar_index: int,
        date: pd.Timestamp,
        comment: str,
    ) -> None:
        if self._position is not None:
            return  # already in a position; TODO: support partial adds

        commission_rate = self._commission_rate
        max_spend = self._cash * min(pct, 1.0)

        if size is not None:
            shares = (size // _LOT) * _LOT
        else:
            cost_per_share = price * (1 + commission_rate)
            shares = math.floor(max_spend / (cost_per_share * _LOT)) * _LOT

        if shares <= 0:
            return

        commission = shares * price * commission_rate
        commission = max(commission, 5.0)  # A-share minimum commission ¥5
        total_cost = shares * price + commission

        if total_cost > self._cash:
            # Scale down
            shares = math.floor((self._cash / (price * (1 + commission_rate))) / _LOT) * _LOT
            if shares <= 0:
                return
            commission = max(shares * price * commission_rate, 5.0)
            total_cost = shares * price + commission

        self._cash -= total_cost
        self._position = Position(
            symbol=self.symbol,
            entry_date=date,
            entry_price=price,
            shares=shares,
            buy_bar=bar_index,
            comment=comment,
        )

    def _execute_sell(
        self,
        price: float,
        bar_index: int,
        date: pd.Timestamp,
        comment: str,
    ) -> None:
        pos = self._position
        if pos is None:
            return

        # T+1 enforcement
        if bar_index <= pos.buy_bar:
            return

        shares = pos.shares
        commission = max(shares * price * self._commission_rate, 5.0)
        stamp_duty = shares * price * self._stamp_duty
        proceeds = shares * price - commission - stamp_duty

        # Costs on entry leg (commission only, no stamp on buy)
        entry_commission = max(shares * pos.entry_price * self._commission_rate, 5.0)
        total_cost = shares * pos.entry_price + entry_commission
        slippage_cost = (
            shares * pos.entry_price * self._slippage
            + shares * price * self._slippage
        )

        pnl = proceeds - total_cost + shares * pos.entry_price  # net
        # Simpler: pnl = net_sell - net_buy
        pnl = (shares * price - commission - stamp_duty) - (shares * pos.entry_price + entry_commission)

        self._trade_counter += 1
        self._trades.append(Trade(
            id=self._trade_counter,
            symbol=self.symbol,
            entry_date=pos.entry_date,
            exit_date=date,
            entry_price=pos.entry_price,
            exit_price=price,
            shares=shares,
            commission=commission + entry_commission,
            slippage=slippage_cost,
            pnl=round(pnl, 4),
            pnl_pct=price / pos.entry_price - 1,
            comment_entry=pos.comment,
            comment_exit=comment,
        ))

        self._cash += proceeds
        self._position = None
