from __future__ import annotations
from typing import Optional
import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

def _require_pandas():
    if pd is None:
        raise RuntimeError("pandas is required. Install with `pip install pandas`.")

def fetch_prices_yf(ticker: str, start: Optional[str] = None, end: Optional[str] = None, interval: str = "1d"):
    """
    Fetch adjusted close prices via yfinance. Returns a pandas Series of Close.
    """
    _require_pandas()
    try:
        import yfinance as yf
    except Exception as e:  # pragma: no cover
        raise RuntimeError("yfinance not installed. `pip install yfinance`.") from e

    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker} in range [{start}, {end}]")
    s = (df["Close"] if "Close" in df.columns else df["Adj Close"]).dropna()
    s.name = ticker.upper()
    return s

def to_simple_returns(prices: np.ndarray) -> np.ndarray:
    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 1 or prices.size < 2:
        raise ValueError("prices must be 1D with length >= 2")
    if (prices <= 0).any():
        raise ValueError("prices must be positive")
    return prices[1:] / prices[:-1] - 1.0

def to_log_returns(prices: np.ndarray) -> np.ndarray:
    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 1 or prices.size < 2:
        raise ValueError("prices must be 1D with length >= 2")
    if (prices <= 0).any():
        raise ValueError("prices must be positive")
    return np.diff(np.log(prices))

def returns_from_prices_series(s_prices, kind: str = "log") -> np.ndarray:
    _require_pandas()
    if kind == "log":
        return to_log_returns(s_prices.to_numpy())
    elif kind == "simple":
        return to_simple_returns(s_prices.to_numpy())
    else:
        raise ValueError("kind must be 'log' or 'simple'")