"""
ashare_bt.engine.position
~~~~~~~~~~~~~~~~~~~~~~~~~
Position and Trade record dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


@dataclass
class Trade:
    """Completed round-trip trade record."""

    id: int
    symbol: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: int
    commission: float          # total commission paid (both legs)
    slippage: float            # total slippage cost
    pnl: float                 # net P&L after costs
    pnl_pct: float             # return (exit/entry - 1)
    comment_entry: str = ""
    comment_exit: str = ""
    holding_days: int = 0

    def __post_init__(self) -> None:
        self.holding_days = (self.exit_date - self.entry_date).days

    def __repr__(self) -> str:
        return (
            f"Trade({self.entry_date.date()} → {self.exit_date.date()}, "
            f"shares={self.shares}, pnl={self.pnl:+.2f}, "
            f"ret={self.pnl_pct:+.2%})"
        )


@dataclass
class Position:
    """Currently open long position."""

    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    buy_bar: int                # bar index when bought (for T+1 enforcement)
    comment: str = ""

    def market_value(self, price: float) -> float:
        return self.shares * price

    def unrealised_pnl(self, price: float) -> float:
        return (price - self.entry_price) * self.shares

    def unrealised_pct(self, price: float) -> float:
        return price / self.entry_price - 1

    def __repr__(self) -> str:
        return (
            f"Position(symbol={self.symbol!r}, shares={self.shares}, "
            f"entry={self.entry_price:.2f} on {self.entry_date.date()})"
        )
