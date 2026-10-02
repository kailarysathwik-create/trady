"""Volatility and dispersion feature transforms.

Calculations are strictly causal:
At timestamp T, standard deviations, realized volatility, and True Range
statistics depend only on observations at or before T.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from trady.features.price import compute_simple_return
from trady.features.registry import FeatureMetadata, registry


def compute_rolling_std(returns: np.ndarray, period: int = 20) -> np.ndarray:
    r"""Compute rolling sample standard deviation of returns.

    Mathematical definition:
        \sigma_{t, k} = \sqrt{\frac{1}{k-1} \sum_{i=0}^{k-1} (r_{t-i} - \bar{r}_t)^2}

    Args:
        returns: 1D array of single-period returns.
        period: Rolling window length k (k >= 2).

    Returns:
        1D float64 array of standard deviations with NaN for indices < period.
    """
    if period < 2:
        raise ValueError(
            f"Period must be at least 2 for standard deviation, got {period}"
        )

    ret_arr = np.asarray(returns, dtype=np.float64)
    n = len(ret_arr)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return result

    # Find valid finite values
    # In sliding window, we require windows of valid returns
    windows = sliding_window_view(ret_arr, period)  # shape: (n - period + 1, period)
    # ddof=1 for unbiased sample standard deviation
    with np.errstate(invalid="ignore"):
        stds = np.nanstd(windows, axis=-1, ddof=1)

    result[period - 1 :] = stds
    return result


def compute_realized_volatility(
    close: np.ndarray,
    period: int = 20,
    periods_per_year: int = 252,
) -> np.ndarray:
    r"""Compute annualized realized volatility from close prices.

    Mathematical definition:
        \text{RV}_{t, k} = \sigma_{t, k}(r_{1d}) \times \sqrt{N_{\text{annual}}}

    Args:
        close: 1D array of close prices.
        period: Rolling return window length (k >= 2).
        periods_per_year: Annualization factor (e.g. 252 for daily trading days).

    Returns:
        1D float64 array of annualized realized volatility with NaN for warmup bars.
    """
    ret = compute_simple_return(close, period=1)
    std = compute_rolling_std(ret, period=period)
    annualization_factor = np.sqrt(periods_per_year)
    return std * annualization_factor


def compute_true_range(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    r"""Compute causal True Range (TR) for each bar.

    Mathematical definition:
        \text{TR}_t = \begin{cases}
            H_0 - L_0 & \text{if } t = 0 \\
            \max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|) & \text{if } t > 0
        \end{cases}

    Args:
        high: 1D array of high prices.
        low: 1D array of low prices.
        close: 1D array of close prices.

    Returns:
        1D float64 array of True Range values.
    """
    high_arr = np.asarray(high, dtype=np.float64)
    low_arr = np.asarray(low, dtype=np.float64)
    close_arr = np.asarray(close, dtype=np.float64)

    n = len(high_arr)
    if n == 0:
        return np.empty(0, dtype=np.float64)

    tr = np.empty(n, dtype=np.float64)
    tr[0] = high_arr[0] - low_arr[0]

    if n > 1:
        prev_close = close_arr[:-1]
        hl = high_arr[1:] - low_arr[1:]
        hc = np.abs(high_arr[1:] - prev_close)
        lc = np.abs(low_arr[1:] - prev_close)
        tr[1:] = np.maximum(hl, np.maximum(hc, lc))

    return tr


def compute_average_true_range(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    r"""Compute Average True Range (ATR) over rolling window.

    Mathematical definition:
        \text{ATR}_{t, k} = \frac{1}{k} \sum_{i=0}^{k-1} \text{TR}_{t-i}

    Args:
        high: 1D array of high prices.
        low: 1D array of low prices.
        close: 1D array of close prices.
        period: Lookback window k (k >= 1).

    Returns:
        1D float64 array of ATR values with NaN for indices < period - 1.
    """
    if period < 1:
        raise ValueError(f"Period must be at least 1, got {period}")

    tr = compute_true_range(high, low, close)
    n = len(tr)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return result

    cumsum = np.cumsum(np.insert(tr, 0, 0.0))
    result[period - 1 :] = (cumsum[period:] - cumsum[:-period]) / period
    return result


def compute_normalized_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    r"""Compute Normalized Average True Range (NATR) as % of close price.

    Mathematical definition:
        NATR_{t, k} = (ATR_{t, k} / C_t) * 100

    Args:
        high: 1D array of high prices.
        low: 1D array of low prices.
        close: 1D array of close prices.
        period: Lookback window k (k >= 1).

    Returns:
        1D float64 array of NATR percentages with NaN for warmup bars.
    """
    atr = compute_average_true_range(high, low, close, period=period)
    close_arr = np.asarray(close, dtype=np.float64)
    result = np.full(len(close_arr), np.nan, dtype=np.float64)

    valid_mask = ~np.isnan(atr) & (close_arr > 0)
    result[valid_mask] = (atr[valid_mask] / close_arr[valid_mask]) * 100.0
    return result


# -------------------------------------------------------------------------
# Feature Registration
# -------------------------------------------------------------------------

DEFAULT_VOL_PERIODS = (10, 20, 60)
DEFAULT_ATR_PERIODS = (14, 21)


def register_volatility_features() -> None:
    """Register all volatility features with the global registry."""
    # Rolling Standard Deviation of Returns
    for p in DEFAULT_VOL_PERIODS:
        name = f"vol_std_{p}d"
        if not registry.contains(name):
            registry.register(
                name=name,
                func=lambda data, p=p: compute_rolling_std(
                    compute_simple_return(data["close"], period=1), period=p
                ),
                metadata=FeatureMetadata(
                    name=name,
                    description=f"Rolling std of returns over {p} bars",
                    feature_group="VOLATILITY",
                    required_columns=("close",),
                    lookback_period=p,
                    min_observations=p + 1,
                    is_point_in_time=True,
                ),
            )

    # Realized Volatility (Annualized)
    for p in DEFAULT_VOL_PERIODS:
        name = f"realized_vol_{p}d"
        if not registry.contains(name):
            registry.register(
                name=name,
                func=lambda data, p=p: compute_realized_volatility(
                    data["close"], period=p
                ),
                metadata=FeatureMetadata(
                    name=name,
                    description=f"Annualized realized volatility over {p} bars",
                    feature_group="VOLATILITY",
                    required_columns=("close",),
                    lookback_period=p,
                    min_observations=p + 1,
                    is_point_in_time=True,
                ),
            )

    # Average True Range (ATR)
    for p in DEFAULT_ATR_PERIODS:
        name = f"atr_{p}d"
        if not registry.contains(name):
            registry.register(
                name=name,
                func=lambda data, p=p: compute_average_true_range(
                    data["high"], data["low"], data["close"], period=p
                ),
                metadata=FeatureMetadata(
                    name=name,
                    description=f"Average True Range over {p} bars",
                    feature_group="VOLATILITY",
                    required_columns=("high", "low", "close"),
                    lookback_period=p,
                    min_observations=p,
                    is_point_in_time=True,
                ),
            )

    # Normalized ATR (NATR)
    for p in DEFAULT_ATR_PERIODS:
        name = f"natr_{p}d"
        if not registry.contains(name):
            registry.register(
                name=name,
                func=lambda data, p=p: compute_normalized_atr(
                    data["high"], data["low"], data["close"], period=p
                ),
                metadata=FeatureMetadata(
                    name=name,
                    description=f"Normalized Average True Range over {p} bars",
                    feature_group="VOLATILITY",
                    required_columns=("high", "low", "close"),
                    lookback_period=p,
                    min_observations=p,
                    is_point_in_time=True,
                ),
            )


# Auto-register on import
register_volatility_features()
