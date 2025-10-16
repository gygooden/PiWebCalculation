from __future__ import annotations
import math
from typing import Optional, Tuple
import numpy as np

# If you want ticker import:
from piwebcalc.data.ingest import fetch_prices_yf, returns_from_prices_series

# -----------------------------
# Your original (slightly polished)
# -----------------------------
def monte_carlo_var(portfolio_returns, num_simulations, confidence_level, seed: Optional[int] = None) -> float:
    if not (0.0 < confidence_level < 1.0):
        raise ValueError("confidence_level must be in (0,1)")
    if seed is not None:
        np.random.seed(seed)
    r = np.asarray(portfolio_returns, dtype=float)
    if r.ndim != 1 or r.size == 0:
        raise ValueError("portfolio_returns must be a non-empty 1D array")
    simulated_returns = np.random.choice(r, size=num_simulations, replace=True)
    losses = -simulated_returns
    return float(np.percentile(losses, (1 - confidence_level) * 100, method="higher"))

# -----------------------------
# Historical VaR / CVaR
# -----------------------------
def historical_var(returns: np.ndarray, confidence_level: float) -> float:
    if not (0.0 < confidence_level < 1.0):
        raise ValueError("confidence_level must be in (0,1)")
    losses = -np.asarray(returns, dtype=float)
    return float(np.percentile(losses, (1.0 - confidence_level) * 100.0, method="higher"))

def historical_cvar(returns: np.ndarray, confidence_level: float) -> float:
    var = historical_var(returns, confidence_level)
    losses = -np.asarray(returns, dtype=float)
    tail = losses[losses >= var]
    return float(tail.mean() if tail.size else var)

# -----------------------------
# Rolling VaR + Backtesting
# -----------------------------
def rolling_historical_var(returns: np.ndarray, window: int, confidence_level: float) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    n = r.size
    out = np.full(n, np.nan, dtype=float)
    for t in range(window, n):
        out[t] = historical_var(r[t - window:t], confidence_level)
    return out

def backtest_var_exceptions(returns: np.ndarray, var_series: np.ndarray) -> Tuple[int, float]:
    r = np.asarray(returns, dtype=float)
    v = np.asarray(var_series, dtype=float)
    if r.shape != v.shape:
        raise ValueError("returns and var_series must have same shape")
    mask = ~np.isnan(v)
    losses = -r[mask]
    v_use = v[mask]
    if v_use.size == 0:
        return 0, float("nan")
    exc = losses > v_use
    k = int(exc.sum())
    return k, k / v_use.size

def kupiec_pof_test(returns: np.ndarray, var_series: np.ndarray, confidence_level: float) -> Tuple[float, int, float]:
    alpha = 1.0 - confidence_level
    r = np.asarray(returns, dtype=float)
    v = np.asarray(var_series, dtype=float)
    mask = ~np.isnan(v)
    r = r[mask]
    v = v[mask]
    if v.size == 0:
        return float("nan"), 0, float("nan")
    losses = -r
    exc = losses > v
    k = int(exc.sum())
    T = v.size
    p_hat = k / T
    eps = 1e-12
    p = min(max(p_hat, eps), 1 - eps)
    logL0 = (T - k) * math.log1p(-alpha) + k * math.log(alpha)
    logL1 = (T - k) * math.log1p(-p) + k * math.log(p)
    LR = -2.0 * (logL0 - logL1)
    return float(LR), k, float(p_hat)

# -----------------------------
# High-level: analyze a ticker
# -----------------------------
def analyze_ticker(
    ticker: str,
    start: Optional[str],
    end: Optional[str],
    confidence_level: float = 0.95,
    window: int = 250,
    return_kind: str = "log",
) -> dict:
    prices = fetch_prices_yf(ticker, start=start, end=end)   # pandas Series
    rets = returns_from_prices_series(prices, kind=return_kind)
    var_series = rolling_historical_var(rets, window=window, confidence_level=confidence_level)
    LR, exceptions, exc_rate = kupiec_pof_test(rets, var_series, confidence_level)

    latest_var = float(var_series[~np.isnan(var_series)][-1]) if np.any(~np.isnan(var_series)) else float("nan")

    # CVaR on the last window for context
    if np.sum(~np.isnan(var_series)) > 0:
        idx_last = np.where(~np.isnan(var_series))[0][-1]
        window_slice = rets[idx_last - window + 1 : idx_last + 1]
        latest_cvar = historical_cvar(window_slice, confidence_level)
    else:
        latest_cvar = float("nan")

    return {
        "ticker": ticker.upper(),
        "date_range": {"start": str(start), "end": str(end)},
        "settings": {"confidence_level": confidence_level, "window": window, "return_kind": return_kind},
        "latest": {"historical_VaR": latest_var, "historical_CVaR": float(latest_cvar)},
        "backtest": {
            "observations_used": int(np.sum(~np.isnan(var_series))),
            "exceptions": int(exceptions),
            "exception_rate": float(exc_rate) if exc_rate == exc_rate else float("nan"),
            "kupiec_LR_stat": float(LR),
            "kupiec_chi2_crit_5pct": 3.84,
        },
    }