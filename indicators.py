"""
ashare_bt.utils.indicators
~~~~~~~~~~~~~~~~~~~~~~~~~~
Vectorised technical indicators that operate on numpy arrays.
All functions return arrays of the same length as the input,
with leading NaNs where the indicator is not yet defined.
"""

from __future__ import annotations

import numpy as np


def sma(closes: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average."""
    out = np.full(len(closes), np.nan)
    for i in range(period - 1, len(closes)):
        out[i] = closes[i - period + 1:i + 1].mean()
    return out


def ema(closes: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average."""
    k = 2.0 / (period + 1)
    out = np.full(len(closes), np.nan)
    out[0] = closes[0]
    for i in range(1, len(closes)):
        prev = out[i - 1] if not np.isnan(out[i - 1]) else closes[i]
        out[i] = closes[i] * k + prev * (1 - k)
    return out


def rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder RSI."""
    out = np.full(len(closes), np.nan)
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()

    if avg_loss == 0:
        out[period] = 100.0
    else:
        out[period] = 100 - 100 / (1 + avg_gain / avg_loss)

    for i in range(period + 1, len(closes)):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        if avg_loss == 0:
            out[i] = 100.0
        else:
            out[i] = 100 - 100 / (1 + avg_gain / avg_loss)
    return out


def macd(
    closes: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MACD indicator.

    Returns
    -------
    macd_line, signal_line, histogram
    """
    fast_ema = ema(closes, fast)
    slow_ema = ema(closes, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(
    closes: np.ndarray,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bollinger Bands.

    Returns
    -------
    upper, middle, lower
    """
    mid = sma(closes, period)
    upper = np.full(len(closes), np.nan)
    lower = np.full(len(closes), np.nan)
    for i in range(period - 1, len(closes)):
        std = closes[i - period + 1:i + 1].std(ddof=0)
        upper[i] = mid[i] + num_std * std
        lower[i] = mid[i] - num_std * std
    return upper, mid, lower


def atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Average True Range."""
    n = len(close)
    tr = np.full(n, np.nan)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    out = np.full(n, np.nan)
    out[period - 1] = tr[:period].mean()
    for i in range(period, n):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


def kdj(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    n: int = 9,
    m1: int = 3,
    m2: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    KDJ indicator (stochastic oscillator variant common in A-share analysis).

    Returns
    -------
    K, D, J
    """
    size = len(close)
    rsv = np.full(size, np.nan)
    for i in range(n - 1, size):
        hh = high[i - n + 1:i + 1].max()
        ll = low[i - n + 1:i + 1].min()
        rsv[i] = (close[i] - ll) / (hh - ll) * 100 if hh != ll else 50.0

    K = np.full(size, np.nan)
    D = np.full(size, np.nan)
    K[n - 1] = 50.0
    D[n - 1] = 50.0
    for i in range(n, size):
        K[i] = (2 / m1) * K[i - 1] + (1 / m1) * rsv[i]
        D[i] = (2 / m2) * D[i - 1] + (1 / m2) * K[i]
    J = 3 * K - 2 * D
    return K, D, J
