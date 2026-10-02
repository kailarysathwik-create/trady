"""Price return feature transforms.

All calculations are strictly causal:
At timestamp T, returns depend only on prices at T and T - period.
"""

import numpy as np

from trady.features.registry import FeatureMetadata, registry


def compute_simple_return(close: np.ndarray, period: int = 1) -> np.ndarray:
    r"""Compute simple percentage return over a given lookback period.

    Mathematical definition:
        R_{t, k} = \frac{P_t - P_{t-k}}{P_{t-k}}

    Args:
        close: 1D array of positive close prices.
        period: Number of periods for lag (k >= 1).

    Returns:
        1D float64 array of returns with NaN for the first `period` elements.
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
    # Causal slice: index t uses close[t] and close[t-period]
    result[period:][valid_mask] = (
        close_arr[period:][valid_mask] - denom[valid_mask]
    ) / denom[valid_mask]
    return result


def compute_log_return(close: np.ndarray, period: int = 1) -> np.ndarray:
    r"""Compute continuously compounded log return over a given lookback period.

    Mathematical definition:
        r_{t, k} = \ln\left(\frac{P_t}{P_{t-k}}\right)

    Args:
        close: 1D array of positive close prices.
        period: Number of periods for lag (k >= 1).

    Returns:
        1D float64 array of log returns with NaN for the first `period` elements.
    """
    if period < 1:
        raise ValueError(f"Period must be at least 1, got {period}")

    close_arr = np.asarray(close, dtype=np.float64)
    n = len(close_arr)
    result = np.full(n, np.nan, dtype=np.float64)
    if n <= period:
        return result

    denom = close_arr[:-period]
    numer = close_arr[period:]
    valid_mask = (denom > 0) & (numer > 0)
    result[period:][valid_mask] = np.log(numer[valid_mask] / denom[valid_mask])
    return result


def compute_multi_period_returns(
    close: np.ndarray,
    periods: tuple[int, ...] = (1, 3, 5, 10, 21),
) -> dict[str, np.ndarray]:
    """Compute simple returns across multiple lookback horizons.

    Args:
        close: 1D array of close prices.
        periods: Sequence of lookback horizons.

    Returns:
        Dictionary mapping feature names (e.g. 'return_1d') to return arrays.
    """
    results: dict[str, np.ndarray] = {}
    for p in periods:
        results[f"return_{p}d"] = compute_simple_return(close, period=p)
    return results


# -------------------------------------------------------------------------
# Feature Registration
# -------------------------------------------------------------------------

DEFAULT_RETURN_PERIODS = (1, 3, 5, 10, 21)


def register_price_features() -> None:
    """Register all price return features with the global registry."""
    # Simple Returns
    for p in DEFAULT_RETURN_PERIODS:
        name = f"return_{p}d"
        if not registry.contains(name):
            registry.register(
                name=name,
                func=lambda data, p=p: compute_simple_return(data["close"], period=p),
                metadata=FeatureMetadata(
                    name=name,
                    description=f"Simple return over {p} period(s)",
                    feature_group="PRICE",
                    required_columns=("close",),
                    lookback_period=p,
                    min_observations=p + 1,
                    is_point_in_time=True,
                ),
            )

    # Log Returns
    for p in (1, 5, 21):
        name = f"log_return_{p}d"
        if not registry.contains(name):
            registry.register(
                name=name,
                func=lambda data, p=p: compute_log_return(data["close"], period=p),
                metadata=FeatureMetadata(
                    name=name,
                    description=f"Log return over {p} period(s): ln(P_t / P_{{t-{p}}})",
                    feature_group="PRICE",
                    required_columns=("close",),
                    lookback_period=p,
                    min_observations=p + 1,
                    is_point_in_time=True,
                ),
            )


# Auto-register on import
register_price_features()
