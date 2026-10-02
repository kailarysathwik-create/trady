"""Forward-looking target calculations and generators for TRADY.

CRITICAL:
These functions compute future outcomes (t > T) for supervised learning.
They must NEVER be included in model input features.
"""

from typing import Literal

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from trady.targets.metadata import TargetMetadata
from trady.targets.registry import target_registry


def compute_forward_return(
    close: np.ndarray,
    horizon: int = 1,
    method: Literal["simple", "log"] = "simple",
) -> np.ndarray:
    r"""Compute future forward return over horizon H.

    Mathematical definition:
        Simple: R^{\text{fwd}}_{t, H} = \frac{P_{t+H} - P_t}{P_t}
        Log:    r^{\text{fwd}}_{t, H} = \ln\left(\frac{P_{t+H}}{P_t}\right)

    For the final H bars, future prices do not exist and are assigned NaN.

    Args:
        close: 1D array of close prices.
        horizon: Future lookahead horizon H (H >= 1).
        method: 'simple' or 'log'.

    Returns:
        1D float64 array of forward returns with NaN for the last H elements.
    """
    if horizon < 1:
        raise ValueError(f"Horizon must be at least 1, got {horizon}")

    close_arr = np.asarray(close, dtype=np.float64)
    n = len(close_arr)
    result = np.full(n, np.nan, dtype=np.float64)
    if n <= horizon:
        return result

    curr_p = close_arr[:-horizon]
    future_p = close_arr[horizon:]

    if method == "simple":
        valid = curr_p > 0
        result[:-horizon][valid] = (future_p[valid] - curr_p[valid]) / curr_p[valid]
    elif method == "log":
        valid = (curr_p > 0) & (future_p > 0)
        result[:-horizon][valid] = np.log(future_p[valid] / curr_p[valid])
    else:
        raise ValueError(f"Unknown return method '{method}'. Use 'simple' or 'log'")

    return result


def compute_binary_classification_target(
    close: np.ndarray,
    horizon: int = 1,
    threshold: float = 0.0,
) -> np.ndarray:
    r"""Compute binary classification target indicating if future return > threshold.

    Mathematical definition:
        Y^{\text{binary}}_{t, H, \theta} = \begin{cases}
            1.0 & \text{if } R^{\text{fwd}}_{t, H} > \theta \\
            0.0 & \text{if } R^{\text{fwd}}_{t, H} \le \theta \\
            \text{NaN} & \text{if } t > N - 1 - H
        \end{cases}

    Args:
        close: 1D array of close prices.
        horizon: Future lookahead horizon H (H >= 1).
        threshold: Return cutoff threshold (default: 0.0).

    Returns:
        1D float64 array of {0.0, 1.0} with NaN for unavailable future rows.
    """
    fwd_ret = compute_forward_return(close, horizon=horizon, method="simple")
    result = np.full(len(close), np.nan, dtype=np.float64)

    valid_mask = ~np.isnan(fwd_ret)
    result[valid_mask] = np.where(fwd_ret[valid_mask] > threshold, 1.0, 0.0)
    return result


def compute_forward_volatility(
    close: np.ndarray,
    horizon: int = 5,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> np.ndarray:
    r"""Compute future realized volatility over horizon H.

    Mathematical definition:
        \sigma^{\text{fwd}}_{t, H} =
            \text{std}\left(r_{t+1 \dots t+H}, \text{ddof}=1\right)
        \text{RV}^{\text{fwd}}_{t, H} =
            \sigma^{\text{fwd}}_{t, H} \times \sqrt{N_{\text{annual}}}

    Requires horizon >= 2. For final H bars, future volatility is NaN.

    Args:
        close: 1D array of close prices.
        horizon: Future window length (H >= 2).
        annualize: Whether to multiply by sqrt(periods_per_year).
        periods_per_year: Trading periods per year (e.g. 252 for daily bars).

    Returns:
        1D float64 array of future realized volatility with NaN for final H elements.
    """
    if horizon < 2:
        raise ValueError(f"Forward volatility requires horizon >= 2, got {horizon}")

    close_arr = np.asarray(close, dtype=np.float64)
    n = len(close_arr)
    result = np.full(n, np.nan, dtype=np.float64)
    if n <= horizon:
        return result

    # 1-day returns for each step: length n - 1
    returns_1d = (close_arr[1:] - close_arr[:-1]) / close_arr[:-1]

    # Sliding window over 1-day returns: shape (n - horizon, horizon)
    future_windows = sliding_window_view(returns_1d, horizon)
    std_vals = np.std(future_windows, axis=-1, ddof=1)

    if annualize:
        std_vals *= np.sqrt(periods_per_year)

    result[: n - horizon] = std_vals
    return result


# -------------------------------------------------------------------------
# Default Target Registration
# -------------------------------------------------------------------------

DEFAULT_RETURN_HORIZONS = (1, 3, 5, 10, 21)
DEFAULT_VOL_HORIZONS = (5, 10, 21)


def register_default_targets() -> None:
    """Register standard research targets in the central target registry."""
    # 1. Forward Simple Returns
    for h in DEFAULT_RETURN_HORIZONS:
        name = f"target_fwd_ret_{h}d"
        if not target_registry.contains(name):
            target_registry.register(
                name=name,
                func=lambda data, h=h: compute_forward_return(
                    data["close"], horizon=h, method="simple"
                ),
                metadata=TargetMetadata(
                    name=name,
                    horizon=h,
                    target_type="continuous",
                    calculation_method="forward_simple_return",
                    description=(
                        f"Future simple return over {h}-bar forward horizon: "
                        f"(P_{{t+{h}}} - P_t) / P_t"
                    ),
                ),
            )

    # 2. Binary Classification Targets (Return > 0.0)
    for h in DEFAULT_RETURN_HORIZONS:
        name = f"target_binary_up_{h}d"
        if not target_registry.contains(name):
            target_registry.register(
                name=name,
                func=lambda data, h=h: compute_binary_classification_target(
                    data["close"], horizon=h, threshold=0.0
                ),
                metadata=TargetMetadata(
                    name=name,
                    horizon=h,
                    target_type="binary",
                    threshold=0.0,
                    calculation_method="binary_threshold",
                    description=(
                        f"Binary indicator whether future {h}-bar return is "
                        "strictly positive (> 0.0)"
                    ),
                ),
            )

    # 3. Binary Classification Targets with Return Threshold (e.g. Return > 1.0%)
    for h in (5, 10):
        name = f"target_binary_gt1pct_{h}d"
        if not target_registry.contains(name):
            target_registry.register(
                name=name,
                func=lambda data, h=h: compute_binary_classification_target(
                    data["close"], horizon=h, threshold=0.01
                ),
                metadata=TargetMetadata(
                    name=name,
                    horizon=h,
                    target_type="binary",
                    threshold=0.01,
                    calculation_method="binary_threshold",
                    description=(
                        f"Binary indicator whether future {h}-bar return "
                        "exceeds +1.0% (> 0.01)"
                    ),
                ),
            )

    # 4. Future Realized Volatility Targets
    for h in DEFAULT_VOL_HORIZONS:
        name = f"target_fwd_vol_{h}d"
        if not target_registry.contains(name):
            target_registry.register(
                name=name,
                func=lambda data, h=h: compute_forward_volatility(
                    data["close"], horizon=h
                ),
                metadata=TargetMetadata(
                    name=name,
                    horizon=h,
                    target_type="continuous",
                    calculation_method="forward_realized_volatility",
                    description=(
                        f"Annualized realized volatility of returns over "
                        f"future {h}-bar window"
                    ),
                ),
            )


# Auto-register on import
register_default_targets()
