"""Volume feature transforms and statistical measures.

Calculations are strictly causal:
At timestamp T, volume changes and rolling statistics depend only on
volume observations recorded at or before T.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from trady.features.momentum import compute_sma
from trady.features.registry import FeatureMetadata, registry


def compute_volume_change(volume: np.ndarray, period: int = 1) -> np.ndarray:
    r"""Compute relative volume percentage change over a given lag.

    Mathematical definition:
        \Delta V_{t, k} = \frac{V_t - V_{t-k}}{V_{t-k}}

    Args:
        volume: 1D array of non-negative volume observations.
        period: Lag period (k >= 1).

    Returns:
        1D float64 array of relative changes with NaN for indices < period.
    """
    if period < 1:
        raise ValueError(f"Period must be at least 1, got {period}")

    vol_arr = np.asarray(volume, dtype=np.float64)
    n = len(vol_arr)
    result = np.full(n, np.nan, dtype=np.float64)
    if n <= period:
        return result

    denom = vol_arr[:-period]
    valid_mask = denom > 0
    result[period:][valid_mask] = (
        vol_arr[period:][valid_mask] - denom[valid_mask]
    ) / denom[valid_mask]
    return result


def compute_volume_to_sma_ratio(volume: np.ndarray, period: int = 20) -> np.ndarray:
    r"""Compute ratio between current volume and its rolling simple moving average.

    Mathematical definition:
        \text{VolRatio}_{t, k} = \frac{V_t}{\text{SMA}_k(V)_t}

    Args:
        volume: 1D array of volume observations.
        period: Lookback window k (k >= 1).

    Returns:
        1D float64 array of volume ratios with NaN for indices < period - 1.
    """
    vol_arr = np.asarray(volume, dtype=np.float64)
    sma = compute_sma(vol_arr, period=period)
    result = np.full(len(vol_arr), np.nan, dtype=np.float64)

    valid_mask = ~np.isnan(sma) & (sma > 0)
    result[valid_mask] = vol_arr[valid_mask] / sma[valid_mask]
    return result


def compute_volume_zscore(
    volume: np.ndarray,
    period: int = 20,
    eps: float = 1e-8,
) -> np.ndarray:
    r"""Compute rolling volume Z-score (standardized volume shock).

    Mathematical definition:
        Z_{V, t, k} = \frac{V_t - \mu_{V, t, k}}{\sigma_{V, t, k} + \epsilon}

    Args:
        volume: 1D array of volume observations.
        period: Lookback window k (k >= 2).
        eps: Small numerical constant to avoid division by zero.

    Returns:
        1D float64 array of Z-scores with NaN for indices < period - 1.
    """
    if period < 2:
        raise ValueError(f"Period must be at least 2 for Z-score, got {period}")

    vol_arr = np.asarray(volume, dtype=np.float64)
    n = len(vol_arr)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return result

    windows = sliding_window_view(vol_arr, period)  # shape: (n - period + 1, period)
    means = np.mean(windows, axis=-1)
    stds = np.std(windows, axis=-1, ddof=1)

    curr_vol = vol_arr[period - 1 :]
    result[period - 1 :] = (curr_vol - means) / (stds + eps)
    return result


# -------------------------------------------------------------------------
# Feature Registration
# -------------------------------------------------------------------------

DEFAULT_VOL_PERIODS = (1, 5)
DEFAULT_VOL_STATS_PERIODS = (10, 20)


def register_volume_features() -> None:
    """Register all volume features with the global registry."""
    # Volume Relative Change
    for p in DEFAULT_VOL_PERIODS:
        name = f"volume_change_{p}d"
        if not registry.contains(name):
            registry.register(
                name=name,
                func=lambda data, p=p: compute_volume_change(data["volume"], period=p),
                metadata=FeatureMetadata(
                    name=name,
                    description=f"Relative volume change over {p} period(s)",
                    feature_group="VOLUME",
                    required_columns=("volume",),
                    lookback_period=p,
                    min_observations=p + 1,
                    is_point_in_time=True,
                ),
            )

    # Volume to SMA Ratio
    for p in DEFAULT_VOL_STATS_PERIODS:
        name = f"volume_to_sma_{p}d"
        if not registry.contains(name):
            registry.register(
                name=name,
                func=lambda data, p=p: compute_volume_to_sma_ratio(
                    data["volume"], period=p
                ),
                metadata=FeatureMetadata(
                    name=name,
                    description=f"Volume to {p}-period volume SMA ratio",
                    feature_group="VOLUME",
                    required_columns=("volume",),
                    lookback_period=p,
                    min_observations=p,
                    is_point_in_time=True,
                ),
            )

    # Volume Z-score
    for p in DEFAULT_VOL_STATS_PERIODS:
        name = f"volume_zscore_{p}d"
        if not registry.contains(name):
            registry.register(
                name=name,
                func=lambda data, p=p: compute_volume_zscore(data["volume"], period=p),
                metadata=FeatureMetadata(
                    name=name,
                    description=f"Rolling volume Z-score over {p} bars",
                    feature_group="VOLUME",
                    required_columns=("volume",),
                    lookback_period=p,
                    min_observations=p,
                    is_point_in_time=True,
                ),
            )


# Auto-register on import
register_volume_features()
