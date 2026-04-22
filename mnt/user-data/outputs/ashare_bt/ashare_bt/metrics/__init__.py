"""
ashare_bt.metrics
~~~~~~~~~~~~~~~~~
Performance metric calculations on an equity curve.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .engine.position import Trade


def compute_metrics(
    equity_curve: pd.Series,
    trades: List[Trade],
    initial_cash: float,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> dict:
    """
    Compute a comprehensive set of performance metrics.

    Parameters
    ----------
    equity_curve : pd.Series
        Portfolio value indexed by date.
    trades : list of Trade
    initial_cash : float
    risk_free_rate : float
        Annual risk-free rate (default 2 % for China).
    periods_per_year : int
        Trading days per year (252 for daily data).

    Returns
    -------
    dict
        Keys follow a consistent naming convention.
    """
    eq = equity_curve.values.astype(float)
    n = len(eq)

    # ── Returns ─────────────────────────────────────────────────────────────────
    daily_rets = np.diff(eq) / eq[:-1]
    total_return = eq[-1] / initial_cash - 1
    years = n / periods_per_year
    ann_return = (eq[-1] / initial_cash) ** (1 / years) - 1 if years > 0 else 0.0

    # ── Risk ────────────────────────────────────────────────────────────────────
    ann_vol = daily_rets.std(ddof=1) * np.sqrt(periods_per_year) if len(daily_rets) > 1 else 0.0

    # Sharpe
    rf_daily = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = daily_rets - rf_daily
    sharpe = (excess.mean() / excess.std(ddof=1) * np.sqrt(periods_per_year)
               if excess.std(ddof=1) > 0 else 0.0)

    # Sortino (downside deviation)
    neg = daily_rets[daily_rets < rf_daily] - rf_daily
    downside_std = np.sqrt((neg ** 2).mean()) * np.sqrt(periods_per_year) if len(neg) > 0 else 0.0
    sortino = (ann_return - risk_free_rate) / downside_std if downside_std > 0 else 0.0

    # Max drawdown
    peak = np.maximum.accumulate(eq)
    drawdown = (eq - peak) / peak
    max_dd = drawdown.min()
    max_dd_pct = max_dd * 100

    # Calmar
    calmar = ann_return / abs(max_dd) if max_dd < 0 else float("inf")

    # ── Drawdown duration ────────────────────────────────────────────────────────
    in_dd = drawdown < 0
    dd_lengths = []
    cur = 0
    for v in in_dd:
        if v:
            cur += 1
        else:
            if cur > 0:
                dd_lengths.append(cur)
            cur = 0
    if cur > 0:
        dd_lengths.append(cur)
    max_dd_duration = max(dd_lengths) if dd_lengths else 0

    # ── Trade stats ──────────────────────────────────────────────────────────────
    n_trades = len(trades)
    if n_trades == 0:
        return {
            "total_return": total_return,
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "max_drawdown": max_dd_pct,
            "max_dd_duration_days": max_dd_duration,
            "n_trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "avg_holding_days": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "total_commission": 0.0,
            "final_equity": float(eq[-1]),
            "initial_cash": initial_cash,
        }

    pnls = np.array([t.pnl for t in trades])
    pnl_pcts = np.array([t.pnl_pct for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    win_rate = len(wins) / n_trades
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0
    gross_profit = wins.sum() if len(wins) else 0.0
    gross_loss = abs(losses.sum()) if len(losses) else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    avg_holding = np.mean([t.holding_days for t in trades])
    total_commission = sum(t.commission for t in trades)

    return {
        "total_return": round(total_return, 6),
        "ann_return": round(ann_return, 6),
        "ann_vol": round(ann_vol, 6),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "calmar": round(calmar, 4),
        "max_drawdown": round(max_dd_pct, 4),
        "max_dd_duration_days": max_dd_duration,
        "n_trades": n_trades,
        "win_rate": round(win_rate, 4),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "profit_factor": round(profit_factor, 4),
        "avg_holding_days": round(avg_holding, 1),
        "best_trade": round(pnl_pcts.max() * 100, 2),
        "worst_trade": round(pnl_pcts.min() * 100, 2),
        "total_commission": round(total_commission, 2),
        "final_equity": round(float(eq[-1]), 2),
        "initial_cash": initial_cash,
    }
