"""
ashare_bt.engine.result
~~~~~~~~~~~~~~~~~~~~~~~
Container for a completed backtest, with reporting and plotting helpers.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
import numpy as np

from position import Trade
from metrics import compute_metrics


class BacktestResult:
    """
    Holds the output of a completed backtest run.

    Attributes
    ----------
    equity_curve : pd.Series
        Portfolio value at the close of each bar.
    trades : list of Trade
        All completed round-trip trades.
    metrics : dict
        Performance statistics (see :func:`~ashare_bt.metrics.compute_metrics`).
    data : pd.DataFrame
        Original OHLCV data (for plotting).
    strategy_name : str
    """

    def __init__(
        self,
        equity_curve: pd.Series,
        trades: List[Trade],
        data: pd.DataFrame,
        strategy_name: str,
        symbol: str,
        initial_cash: float,
        risk_free_rate: float = 0.02,
    ) -> None:
        self.equity_curve = equity_curve
        self.trades = trades
        self.data = data
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.initial_cash = initial_cash
        self.metrics = compute_metrics(equity_curve, trades, initial_cash, risk_free_rate)

    # ── Display ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        m = self.metrics
        return (
            f"BacktestResult(\n"
            f"  strategy    : {self.strategy_name}\n"
            f"  symbol      : {self.symbol}\n"
            f"  period      : {self.equity_curve.index[0].date()} → "
            f"{self.equity_curve.index[-1].date()}\n"
            f"  total_return: {m['total_return']:+.2%}\n"
            f"  ann_return  : {m['ann_return']:+.2%}\n"
            f"  sharpe      : {m['sharpe']:.3f}\n"
            f"  max_drawdown: {m['max_drawdown']:.2f}%\n"
            f"  n_trades    : {m['n_trades']}\n"
            f"  win_rate    : {m['win_rate']:.1%}\n"
            f")"
        )

    def summary(self) -> pd.Series:
        """Return metrics as a labelled pandas Series."""
        m = self.metrics
        labels = {
            "总收益率":    f"{m['total_return']:+.2%}",
            "年化收益率":  f"{m['ann_return']:+.2%}",
            "年化波动率":  f"{m['ann_vol']:.2%}",
            "Sharpe 比率": f"{m['sharpe']:.3f}",
            "Sortino 比率":f"{m['sortino']:.3f}",
            "Calmar 比率": f"{m['calmar']:.3f}",
            "最大回撤":    f"{m['max_drawdown']:.2f}%",
            "最长回撤天数": f"{m['max_dd_duration_days']} days",
            "交易次数":    m['n_trades'],
            "胜率":        f"{m['win_rate']:.1%}",
            "盈亏比":      f"{m['profit_factor']:.2f}",
            "平均持仓天数": f"{m['avg_holding_days']:.1f}",
            "最佳单笔":    f"{m['best_trade']:+.2f}%",
            "最差单笔":    f"{m['worst_trade']:+.2f}%",
            "总手续费":    f"¥{m['total_commission']:,.2f}",
            "期末资产":    f"¥{m['final_equity']:,.2f}",
        }
        return pd.Series(labels, name="value")

    def trades_df(self) -> pd.DataFrame:
        """Return trade list as a tidy DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        rows = []
        for t in self.trades:
            rows.append({
                "id": t.id,
                "entry_date": t.entry_date,
                "exit_date": t.exit_date,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "shares": t.shares,
                "pnl": t.pnl,
                "return_pct": t.pnl_pct * 100,
                "holding_days": t.holding_days,
                "commission": t.commission,
                "comment_entry": t.comment_entry,
                "comment_exit": t.comment_exit,
            })
        df = pd.DataFrame(rows).set_index("id")
        df["cumulative_pnl"] = df["pnl"].cumsum()
        return df

    def drawdown_series(self) -> pd.Series:
        """Percentage drawdown series."""
        eq = self.equity_curve.values
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak * 100
        return pd.Series(dd, index=self.equity_curve.index, name="drawdown_pct")

    # ── Plotting ─────────────────────────────────────────────────────────────────

    def plot(
        self,
        benchmark: Optional[pd.Series] = None,
        figsize: tuple = (14, 10),
        show: bool = True,
    ) -> "matplotlib.figure.Figure":
        """
        Plot equity curve, drawdown, and trade annotations.

        Parameters
        ----------
        benchmark : pd.Series, optional
            External benchmark equity curve indexed by date.
            If None, a buy-and-hold benchmark is computed automatically.
        figsize : tuple
        show : bool
            Call ``plt.show()`` automatically.

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            from matplotlib.ticker import FuncFormatter
        except ImportError:
            raise ImportError("matplotlib is required for plotting: pip install matplotlib")

        # Build buy-and-hold benchmark if not supplied
        if benchmark is None:
            bh = self.data["close"] / self.data["close"].iloc[0] * self.initial_cash
            benchmark = bh.rename("Buy & Hold")

        fig = plt.figure(figsize=figsize, facecolor="#0f1520")
        gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.08)

        ax1 = fig.add_subplot(gs[0])  # equity
        ax2 = fig.add_subplot(gs[1], sharex=ax1)  # drawdown
        ax3 = fig.add_subplot(gs[2], sharex=ax1)  # volume / bar P&L

        for ax in (ax1, ax2, ax3):
            ax.set_facecolor("#0a0e17")
            ax.tick_params(colors="#6b8299", labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#1e2d45")
            ax.grid(color="#1e2d45", linewidth=0.5, linestyle="--")

        # ── equity curve ────────────────────────────────────────────────────────
        ax1.plot(self.equity_curve.index, self.equity_curve.values,
                 color="#f0b429", linewidth=1.5, label=f"Strategy ({self.strategy_name})", zorder=3)
        ax1.plot(benchmark.index, benchmark.values,
                 color="#4da6ff", linewidth=1.0, linestyle="--", alpha=0.7,
                 label=benchmark.name, zorder=2)

        # Trade markers
        tdf = self.trades_df()
        if not tdf.empty:
            buys = tdf["entry_date"]
            sells = tdf["exit_date"]
            # Map dates to equity values
            eq_idx = self.equity_curve.index
            for _, row in tdf.iterrows():
                try:
                    buy_y = self.equity_curve.asof(row["entry_date"])
                    sell_y = self.equity_curve.asof(row["exit_date"])
                    ax1.scatter(row["entry_date"], buy_y, marker="^", color="#00d68f", s=30, zorder=5)
                    ax1.scatter(row["exit_date"], sell_y, marker="v", color="#ff4d6d", s=30, zorder=5)
                except Exception:
                    pass

        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"¥{x/1e4:.1f}w"))
        ax1.set_title(
            f"{self.symbol} · {self.strategy_name} · "
            f"{self.equity_curve.index[0].date()} → {self.equity_curve.index[-1].date()}",
            color="#c8d8e8", fontsize=11, pad=10
        )
        ax1.legend(facecolor="#0f1520", edgecolor="#1e2d45", labelcolor="#c8d8e8", fontsize=8)

        # ── drawdown ─────────────────────────────────────────────────────────────
        dd = self.drawdown_series()
        ax2.fill_between(dd.index, dd.values, 0, color="#ff4d6d", alpha=0.3)
        ax2.plot(dd.index, dd.values, color="#ff4d6d", linewidth=0.8)
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax2.set_ylabel("Drawdown", color="#6b8299", fontsize=8)

        # ── per-trade P&L bars ───────────────────────────────────────────────────
        if not tdf.empty:
            colors = ["#00d68f" if v > 0 else "#ff4d6d" for v in tdf["pnl"]]
            ax3.bar(tdf["exit_date"], tdf["pnl"], color=colors, width=2, alpha=0.8)
        ax3.axhline(0, color="#1e2d45", linewidth=0.5)
        ax3.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"¥{x:.0f}"))
        ax3.set_ylabel("Trade P&L", color="#6b8299", fontsize=8)

        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)

        if show:
            plt.tight_layout()
            plt.show()

        return fig
