"""Price structure and channel position feature transforms.

Calculations are strictly causal:
At timestamp T, distances to rolling highs/lows and range positions depend
only on price bars recorded at or before T.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from trady.features.registry import FeatureMetadata, registry


def compute_distance_from_rolling_high(
    close: np.ndarray,
    high: np.ndarray,
    period: int = 20,
) -> np.ndarray:
    r"""Compute relative distance between close price and rolling highest high.

    Mathematical definition:
        DistHigh_{t, k} = (P_t - max_{0 <= i < k} H_{t-i}) / max_{0 <= i < k} H_{t-i}

    Always non-positive (<= 0.0) as Close_t <= High_t <= RollingHigh_t.

    Args:
        close: 1D array of close prices.
        high: 1D array of high prices.
        period: Lookback window k (k >= 1).

    Returns:
        1D float64 array of relative distances with NaN for indices < period - 1.
    """
    if period < 1:
        raise ValueError(f"Period must be at least 1, got {period}")

    close_arr = np.asarray(close, dtype=np.float64)
    high_arr = np.asarray(high, dtype=np.float64)
    n = len(close_arr)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return result

    windows = sliding_window_view(high_arr, period)
    rolling_max = np.max(windows, axis=-1)

    curr_close = close_arr[period - 1 :]
    valid_mask = rolling_max > 0
    result[period - 1 :][valid_mask] = (
        curr_close[valid_mask] - rolling_max[valid_mask]
    ) / rolling_max[valid_mask]
    return result


def compute_distance_from_rolling_low(
    close: np.ndarray,
    low: np.ndarray,
    period: int = 20,
) -> np.ndarray:
    r"""Compute relative distance between close price and rolling lowest low.

    Mathematical definition:
        DistLow_{t, k} = (P_t - min_{0 <= i < k} L_{t-i}) / min_{0 <= i < k} L_{t-i}

    Always non-negative (>= 0.0) as Close_t >= Low_t >= RollingLow_t.

    Args:
        close: 1D array of close prices.
        low: 1D array of low prices.
        period: Lookback window k (k >= 1).

    Returns:
        1D float64 array of relative distances with NaN for indices < period - 1.
    """
    if period < 1:
        raise ValueError(f"Period must be at least 1, got {period}")

    close_arr = np.asarray(close, dtype=np.float64)
    low_arr = np.asarray(low, dtype=np.float64)
    n = len(close_arr)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return result

    windows = sliding_window_view(low_arr, period)
    rolling_min = np.min(windows, axis=-1)

    curr_close = close_arr[period - 1 :]
    valid_mask = rolling_min > 0
    result[period - 1 :][valid_mask] = (
        curr_close[valid_mask] - rolling_min[valid_mask]
    ) / rolling_min[valid_mask]
    return result


def compute_normalized_range_position(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    period: int = 20,
    eps: float = 1e-8,
) -> np.ndarray:
    r"""Compute normalized position of close price within its rolling [Low, High] range.

    Mathematical definition:
        PosRange_{t, k} = (P_t - min L) / (max H - min L) in [0, 1]

    Analogous to Stochastic %K indicator over k bars.

    Args:
        close: 1D array of close prices.
        high: 1D array of high prices.
        low: 1D array of low prices.
        period: Lookback window k (k >= 1).
        eps: Small constant to avoid zero-division when high == low.

    Returns:
        1D float64 array bounded in [0.0, 1.0] with NaN for indices < period - 1.
    """
    if period < 1:
        raise ValueError(f"Period must be at least 1, got {period}")

    close_arr = np.asarray(close, dtype=np.float64)
    high_arr = np.asarray(high, dtype=np.float64)
    low_arr = np.asarray(low, dtype=np.float64)
    n = len(close_arr)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return result

    high_windows = sliding_window_view(high_arr, period)
    low_windows = sliding_window_view(low_arr, period)
    rolling_max = np.max(high_windows, axis=-1)
    rolling_min = np.min(low_windows, axis=-1)

    curr_close = close_arr[period - 1 :]
    span = rolling_max - rolling_min

    # Where span is flat, default to 0.5 (middle of range)
    pos = np.where(span > eps, (curr_close - rolling_min) / span, 0.5)
    result[period - 1 :] = np.clip(pos, 0.0, 1.0)
    return result


# -------------------------------------------------------------------------
# Feature Registration
# -------------------------------------------------------------------------

DEFAULT_STRUCTURE_PERIODS = (10, 20, 60)


def register_structure_features() -> None:
    """Register all price structure features with the global registry."""
    # Distance from rolling high
    for p in DEFAULT_STRUCTURE_PERIODS:
        name = f"dist_high_{p}d"
        if not registry.contains(name):
            registry.register(
                name=name,
                func=lambda data, p=p: compute_distance_from_rolling_high(
                    data["close"], data["high"], period=p
                ),
                metadata=FeatureMetadata(
                    name=name,
                    description=f"Distance from {p}-bar rolling highest high",
                    feature_group="PRICE_STRUCTURE",
                    required_columns=("close", "high"),
                    lookback_period=p,
                    min_observations=p,
                    is_point_in_time=True,
                ),
            )

    # Distance from rolling low
    for p in DEFAULT_STRUCTURE_PERIODS:
        name = f"dist_low_{p}d"
        if not registry.contains(name):
            registry.register(
                name=name,
                func=lambda data, p=p: compute_distance_from_rolling_low(
                    data["close"], data["low"], period=p
                ),
                metadata=FeatureMetadata(
                    name=name,
                    description=f"Distance from {p}-bar rolling lowest low",
                    feature_group="PRICE_STRUCTURE",
                    required_columns=("close", "low"),
                    lookback_period=p,
                    min_observations=p,
                    is_point_in_time=True,
                ),
            )

    # Normalized range position
    for p in DEFAULT_STRUCTURE_PERIODS:
        name = f"range_pos_{p}d"
        if not registry.contains(name):
            registry.register(
                name=name,
                func=lambda data, p=p: compute_normalized_range_position(
                    data["close"], data["high"], data["low"], period=p
                ),
                metadata=FeatureMetadata(
                    name=name,
                    description=f"Normalized position within {p}-bar rolling range",
                    feature_group="PRICE_STRUCTURE",
                    required_columns=("close", "high", "low"),
                    lookback_period=p,
                    min_observations=p,
                    is_point_in_time=True,
                ),
            )


# Auto-register on import
register_structure_features()
