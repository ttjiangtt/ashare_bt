"""
ashare_bt.engine.backtest
~~~~~~~~~~~~~~~~~~~~~~~~~
Primary entry point for running a backtest.

Usage
-----
::

    from ashare_bt import Backtest, DataFeed
    from ashare_bt.strategy import SMACross

    feed = DataFeed("600519.csv", symbol="600519")

    bt = Backtest(
        data=feed,
        strategy=SMACross,
        cash=100_000,
        commission=0.0003,
        stamp_duty=0.001,
        slippage=0.0005,
        fast=5,
        slow=20,
    )

    result = bt.run()
    print(result)
    print(result.summary())
    result.plot()

    # Optimise
    grid = bt.optimise(fast=range(3, 15), slow=range(10, 40, 5))
    print(grid.sort_values("sharpe", ascending=False).head())
"""

from __future__ import annotations

import copy
import itertools
import warnings
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import pandas as pd

from broker import Broker
from result import BacktestResult
from feed import DataFeed
from base import Strategy


class Backtest:
    """
    Orchestrates a single-asset backtest.

    Parameters
    ----------
    data : DataFeed | pd.DataFrame
        Price data.  If a DataFrame is passed, it will be wrapped in a
        :class:`~ashare_bt.data.feed.DataFeed` automatically.
    strategy : type
        A subclass of :class:`~ashare_bt.strategy.base.Strategy`
        (*not* an instance).
    cash : float
        Starting capital in CNY.
    commission : float
        Per-side commission rate (e.g. ``0.0003`` = 0.03 %).
    stamp_duty : float
        Sell-side stamp duty (default ``0.001`` = 0.1 %).
    slippage : float
        Fractional slippage per order (default ``0.0005``).
    price_limit : float
        Daily price move limit for circuit-breaker rejection (default ``0.10``).
    risk_free_rate : float
        Annual risk-free rate used in Sharpe/Sortino calculation.
    **strategy_params
        Keyword arguments forwarded to the strategy as class attributes,
        overriding the strategy's defaults.
    """

    def __init__(
        self,
        data: Union[DataFeed, pd.DataFrame],
        strategy: Type[Strategy],
        cash: float = 100_000,
        commission: float = 0.0003,
        stamp_duty: float = 0.001,
        slippage: float = 0.0005,
        price_limit: float = 0.10,
        risk_free_rate: float = 0.02,
        **strategy_params: Any,
    ) -> None:
        if isinstance(data, pd.DataFrame):
            data = DataFeed(data)
        self._feed = data
        self._strategy_cls = strategy
        self._cash = cash
        self._commission = commission
        self._stamp_duty = stamp_duty
        self._slippage = slippage
        self._price_limit = price_limit
        self._risk_free_rate = risk_free_rate
        self._strategy_params = strategy_params

    # ── main run ────────────────────────────────────────────────────────────────

    def run(self) -> BacktestResult:
        """
        Execute the backtest and return a :class:`~ashare_bt.engine.result.BacktestResult`.
        """
        feed = copy.deepcopy(self._feed)
        broker = Broker(
            cash=self._cash,
            commission_rate=self._commission,
            stamp_duty=self._stamp_duty,
            slippage=self._slippage,
            price_limit=self._price_limit,
            symbol=feed.symbol,
        )

        # Build strategy instance with overridden params
        strat = self._build_strategy(feed, broker)
        strat.init()

        equity_values: list[float] = []
        n = len(feed)

        for i in range(n):
            feed._cursor = i
            bar = feed.data.iloc[i]

            # Fill any pending orders at today's open
            broker.process_pending(bar, bar_index=i)

            # Run strategy logic with current bar's close available
            strat.next()

            # Record equity at close
            equity_values.append(broker.equity(feed.close))

            # End of day: update prev_close for next bar's limit check
            broker.mark_eod(feed.close)

        equity_curve = pd.Series(
            equity_values,
            index=feed.data.index,
            name="equity",
        )

        return BacktestResult(
            equity_curve=equity_curve,
            trades=broker.trades,
            data=feed.data,
            strategy_name=self._strategy_cls.__name__,
            symbol=feed.symbol,
            initial_cash=self._cash,
            risk_free_rate=self._risk_free_rate,
        )

    # ── optimisation ─────────────────────────────────────────────────────────────

    def optimise(
        self,
        target: str = "sharpe",
        **param_grid: Any,
    ) -> pd.DataFrame:
        """
        Grid-search over strategy parameters.

        Parameters
        ----------
        target : str
            Metric to report (any key in ``result.metrics``).
        **param_grid
            Keyword args mapping parameter names to iterables of values.

            Example::

                bt.optimise(fast=range(3, 15), slow=range(10, 40, 5))

        Returns
        -------
        pd.DataFrame
            One row per parameter combination, sorted by *target* descending.
        """
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combos = list(itertools.product(*values))

        rows = []
        for combo in combos:
            params = dict(zip(keys, combo))
            # Validate: skip if fast >= slow (for MA strategies)
            if "fast" in params and "slow" in params:
                if params["fast"] >= params["slow"]:
                    continue
            try:
                bt = Backtest(
                    data=self._feed,
                    strategy=self._strategy_cls,
                    cash=self._cash,
                    commission=self._commission,
                    stamp_duty=self._stamp_duty,
                    slippage=self._slippage,
                    price_limit=self._price_limit,
                    risk_free_rate=self._risk_free_rate,
                    **{**self._strategy_params, **params},
                )
                result = bt.run()
                row = {**params, **result.metrics}
                rows.append(row)
            except Exception as e:
                warnings.warn(f"Optimisation combo {params} failed: {e}")

        df = pd.DataFrame(rows)
        if target in df.columns:
            df = df.sort_values(target, ascending=False).reset_index(drop=True)
        return df

    # ── helpers ──────────────────────────────────────────────────────────────────

    def _build_strategy(self, feed: DataFeed, broker: Broker) -> Strategy:
        """Instantiate strategy and inject runtime dependencies."""
        # Create a subclass that overrides class-level defaults with our params
        if self._strategy_params:
            strat_cls = type(
                self._strategy_cls.__name__,
                (self._strategy_cls,),
                dict(self._strategy_params),
            )
        else:
            strat_cls = self._strategy_cls

        strat = strat_cls.__new__(strat_cls)
        strat._inject(broker, feed)
        return strat

    def __repr__(self) -> str:
        return (
            f"Backtest(strategy={self._strategy_cls.__name__}, "
            f"symbol={self._feed.symbol!r}, cash={self._cash:,})"
        )
