"""
ashare_bt — A lightweight A-share backtesting library.

Quick start::

    from ashare_bt import Backtest, Strategy
    from ashare_bt.strategy import SMACross

    bt = Backtest(data, SMACross, fast=5, slow=20, cash=100_000)
    result = bt.run()
    print(result.metrics)
    result.plot()
"""

from .engine.backtest import Backtest
from .engine.result import BacktestResult
from .strategy.base import Strategy
from .data.feed import DataFeed

__version__ = "0.1.0"
__all__ = ["Backtest", "BacktestResult", "Strategy", "DataFeed"]
