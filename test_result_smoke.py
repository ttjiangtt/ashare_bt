import pandas as pd
from result import BacktestResult

# Minimal equity series
idx = pd.to_datetime(["2020-01-01","2020-01-02","2020-01-03"])
eq = pd.Series([100000.0, 100100.0, 100200.0], index=idx)

# No trades
trades = []

# Minimal data DataFrame with close column
data = pd.DataFrame({"close":[100000.0,100100.0,100200.0]}, index=idx)

res = BacktestResult(equity_curve=eq, trades=trades, data=data, strategy_name="SMACross", symbol="TEST", initial_cash=100000.0)
print(res.metrics)

