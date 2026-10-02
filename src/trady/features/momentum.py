"""Momentum and trend-relationship feature transforms.

Calculations are strictly causal:
At timestamp T, rolling momentum and moving-average relationships depend
only on observations at or before T.
"""

import numpy as np

from trady.features.registry import FeatureMetadata, registry


def compute_sma(arr: np.ndarray, period: int) -> np.ndarray:
    """Compute strictly causal Simple Moving Average (SMA).

    Mathematical definition:
        SMA_{t, k} = \frac{1}{k} \\sum_{i=0}^{k-1} x_{t-i}

    Args:
        arr: 1D numerical array.
        period: Rolling window length k (k >= 1).

    Returns:
        1D float64 array of SMA values with NaN for indices < period - 1.
    """
    if period < 1:
        raise ValueError(f"Period must be at least 1, got {period}")

    arr_val = np.asarray(arr, dtype=np.float64)
    n = len(arr_val)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return result

    # O(N) causal rolling sum using cumulative sum
    cumsum = np.cumsum(np.insert(arr_val, 0, 0.0))
    result[period - 1 :] = (cumsum[period:] - cumsum[:-period]) / period
    return result


def compute_rolling_momentum(close: np.ndarray, period: int = 10) -> np.ndarray:
    r"""Compute rolling price momentum (difference).

    Mathematical definition:
        M_{t, k} = P_t - P_{t-k}

    Args:
        close: 1D array of close prices.
        period: Number of periods for momentum lag (k >= 1).

    Returns:
        1D float64 array of momentum differences with NaN for indices < period.
    """
    if period < 1:
        raise ValueError(f"Period must be at least 1, got {period}")

    close_arr = np.asarray(close, dtype=np.float64)
    n = len(close_arr)
    result = np.full(n, np.nan, dtype=np.float64)
    if n <= period:
        return result

    result[period:] = close_arr[period:] - close_arr[:-period]
    return result


def compute_rate_of_change(close: np.ndarray, period: int = 10) -> np.ndarray:
    r"""Compute Rate of Change (ROC) expressed as percentage.

    Mathematical definition:
        ROC_{t, k} = \frac{P_t - P_{t-k}}{P_{t-k}} \times 100

    Args:
        close: 1D array of close prices.
        period: Lookback horizon (k >= 1).

    Returns:
        1D float64 array of ROC percentages with NaN for indices < period.
    """
    if period < 1:
        raise ValueError(f"Period must be at least 1, got {period}")

    close_arr = np.asarray(close, dtype=np.float64)
    n = len(close_arr)
    result = np.full(n, np.nan, dtype=np.float64)
    if n <= period:
        return result

    denom = close_arr[:-period]
    valid_mask = denom > 0
    result[period:][valid_mask] = (
        (close_arr[period:][valid_mask] - denom[valid_mask]) / denom[valid_mask]
    ) * 100.0
    return result


def compute_price_to_sma_ratio(close: np.ndarray, period: int = 20) -> np.ndarray:
    r"""Compute percentage distance between close price and its rolling SMA.

    Mathematical definition:
        \text{DistSMA}_{t, k} = \frac{P_t - \text{SMA}_k(P)_t}{\text{SMA}_k(P)_t}

    Args:
        close: 1D array of close prices.
        period: SMA lookback window (k >= 1).

    Returns:
        1D float64 array of relative distances with NaN for indices < period - 1.
    """
    close_arr = np.asarray(close, dtype=np.float64)
    sma = compute_sma(close_arr, period=period)
    result = np.full(len(close_arr), np.nan, dtype=np.float64)

    valid_mask = ~np.isnan(sma) & (sma > 0)
    result[valid_mask] = (close_arr[valid_mask] - sma[valid_mask]) / sma[valid_mask]
    return result


def compute_sma_ratio(
    close: np.ndarray,
    fast_period: int = 10,
    slow_period: int = 50,
) -> np.ndarray:
    r"""Compute relative spread between fast and slow moving averages.

    Mathematical definition:
        SMASpread_t = (SMA_{fast, t} / SMA_{slow, t}) - 1

    Args:
        close: 1D array of close prices.
        fast_period: Window for fast moving average.
        slow_period: Window for slow moving average (must be > fast_period).

    Returns:
        1D float64 array of relative spread with NaN for indices < slow_period - 1.
    """
    if fast_period >= slow_period:
        msg = f"fast_period ({fast_period}) must be < slow_period ({slow_period})"
        raise ValueError(msg)

    close_arr = np.asarray(close, dtype=np.float64)
    fast_sma = compute_sma(close_arr, period=fast_period)
    slow_sma = compute_sma(close_arr, period=slow_period)

    result = np.full(len(close_arr), np.nan, dtype=np.float64)
    valid_mask = ~np.isnan(slow_sma) & (slow_sma > 0) & ~np.isnan(fast_sma)
    result[valid_mask] = (fast_sma[valid_mask] - slow_sma[valid_mask]) / slow_sma[
        valid_mask
    ]
    return result


# -------------------------------------------------------------------------
# Feature Registration
# -------------------------------------------------------------------------

DEFAULT_MOMENTUM_PERIODS = (5, 10, 21)
DEFAULT_SMA_PERIODS = (10, 20, 50)


def register_momentum_features() -> None:
    """Register all momentum features with the global registry."""
    # Rolling Momentum
    for p in DEFAULT_MOMENTUM_PERIODS:
        name = f"momentum_{p}d"
        if not registry.contains(name):
            registry.register(
                name=name,
                func=lambda data, p=p: compute_rolling_momentum(
                    data["close"], period=p
                ),
                metadata=FeatureMetadata(
                    name=name,
                    description=f"Rolling momentum difference over {p} period(s)",
                    feature_group="MOMENTUM",
                    required_columns=("close",),
                    lookback_period=p,
                    min_observations=p + 1,
                    is_point_in_time=True,
                ),
            )

    # Rate of Change (ROC)
    for p in DEFAULT_MOMENTUM_PERIODS:
        name = f"roc_{p}d"
        if not registry.contains(name):
            registry.register(
                name=name,
                func=lambda data, p=p: compute_rate_of_change(data["close"], period=p),
                metadata=FeatureMetadata(
                    name=name,
                    description=f"Rate of change percentage over {p} period(s)",
                    feature_group="MOMENTUM",
                    required_columns=("close",),
                    lookback_period=p,
                    min_observations=p + 1,
                    is_point_in_time=True,
                ),
            )

    # Price to SMA Ratio
    for p in DEFAULT_SMA_PERIODS:
        name = f"price_to_sma_{p}d"
        if not registry.contains(name):
            registry.register(
                name=name,
                func=lambda data, p=p: compute_price_to_sma_ratio(
                    data["close"], period=p
                ),
                metadata=FeatureMetadata(
                    name=name,
                    description=f"Distance from {p}-period SMA",
                    feature_group="MOMENTUM",
                    required_columns=("close",),
                    lookback_period=p,
                    min_observations=p,
                    is_point_in_time=True,
                ),
            )

    # SMA Ratio (Fast / Slow)
    pair = (10, 50)
    name = f"sma_ratio_{pair[0]}_{pair[1]}"
    if not registry.contains(name):
        registry.register(
            name=name,
            func=lambda data, f=pair[0], s=pair[1]: compute_sma_ratio(
                data["close"], fast_period=f, slow_period=s
            ),
            metadata=FeatureMetadata(
                name=name,
                description=f"Spread between {pair[0]}d and {pair[1]}d SMA",
                feature_group="MOMENTUM",
                required_columns=("close",),
                lookback_period=pair[1],
                min_observations=pair[1],
                is_point_in_time=True,
            ),
        )


# Auto-register on import
register_momentum_features()
