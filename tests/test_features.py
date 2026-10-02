"""Unit and causal lookahead-leakage tests for the TRADY Feature Engine."""

import math
from pathlib import Path

import numpy as np
import pytest

from trady.data.provider import SyntheticDataProvider
from trady.features.config import FeatureConfig
from trady.features.momentum import (
    compute_price_to_sma_ratio,
    compute_rate_of_change,
    compute_rolling_momentum,
    compute_sma,
    compute_sma_ratio,
)
from trady.features.pipeline import FeaturePipeline
from trady.features.price import (
    compute_log_return,
    compute_multi_period_returns,
    compute_simple_return,
)
from trady.features.registry import FeatureMetadata, registry
from trady.features.structure import (
    compute_distance_from_rolling_high,
    compute_distance_from_rolling_low,
    compute_normalized_range_position,
)
from trady.features.volatility import (
    compute_average_true_range,
    compute_normalized_atr,
    compute_realized_volatility,
    compute_rolling_std,
    compute_true_range,
)
from trady.features.volume import (
    compute_volume_change,
    compute_volume_to_sma_ratio,
    compute_volume_zscore,
)

# -------------------------------------------------------------------------
# 1. Registry & Metadata Tests
# -------------------------------------------------------------------------


def test_registry_contains_all_feature_groups() -> None:
    """Ensure all required feature groups are present and registered."""
    groups = {m.feature_group for m in registry.list_features()}
    assert "PRICE" in groups
    assert "MOMENTUM" in groups
    assert "VOLATILITY" in groups
    assert "VOLUME" in groups
    assert "PRICE_STRUCTURE" in groups


def test_all_features_declare_point_in_time_safety() -> None:
    """Every registered feature must declare point-in-time safety."""
    for meta in registry.list_features():
        assert meta.is_point_in_time is True
        assert meta.lookback_period >= 1
        assert len(meta.required_columns) >= 1
        assert meta.min_observations >= 1


def test_feature_metadata_validation() -> None:
    """Verify FeatureMetadata rejects zero lookback and empty names."""
    with pytest.raises(ValueError, match="Feature name cannot be empty"):
        FeatureMetadata(name="", description="Test", lookback_period=5)

    with pytest.raises(ValueError, match="at least 1 bar"):
        FeatureMetadata(name="TEST", description="Test", lookback_period=0)

    with pytest.raises(ValueError, match="min_observations"):
        FeatureMetadata(
            name="TEST", description="Test", lookback_period=5, min_observations=0
        )


# -------------------------------------------------------------------------
# 2. Price Return Tests
# -------------------------------------------------------------------------


def test_price_simple_and_log_returns() -> None:
    """Test mathematical correctness of simple and log returns."""
    prices = np.array([100.0, 110.0, 121.0, 108.9])

    simple = compute_simple_return(prices, period=1)
    assert np.isnan(simple[0])
    assert math.isclose(simple[1], 0.10, rel_tol=1e-5)
    assert math.isclose(simple[2], 0.10, rel_tol=1e-5)
    assert math.isclose(simple[3], -0.10, rel_tol=1e-5)

    log_ret = compute_log_return(prices, period=1)
    assert np.isnan(log_ret[0])
    assert math.isclose(log_ret[1], math.log(1.10), rel_tol=1e-5)
    assert math.isclose(log_ret[2], math.log(1.10), rel_tol=1e-5)

    multi = compute_multi_period_returns(prices, periods=(1, 2))
    assert "return_1d" in multi
    assert "return_2d" in multi
    assert math.isclose(multi["return_2d"][2], 0.21, rel_tol=1e-5)


# -------------------------------------------------------------------------
# 3. Momentum Feature Tests
# -------------------------------------------------------------------------


def test_momentum_features_math() -> None:
    """Test SMA, rolling momentum, ROC, and SMA ratios."""
    series = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

    sma3 = compute_sma(series, period=3)
    assert np.isnan(sma3[0])
    assert np.isnan(sma3[1])
    assert math.isclose(sma3[2], 20.0)  # (10+20+30)/3
    assert math.isclose(sma3[3], 30.0)  # (20+30+40)/3
    assert math.isclose(sma3[4], 40.0)  # (30+40+50)/3

    mom2 = compute_rolling_momentum(series, period=2)
    assert np.isnan(mom2[0])
    assert np.isnan(mom2[1])
    assert math.isclose(mom2[2], 20.0)  # 30 - 10
    assert math.isclose(mom2[4], 20.0)  # 50 - 30

    roc2 = compute_rate_of_change(series, period=2)
    assert math.isclose(roc2[2], 200.0)  # (30-10)/10 * 100

    p_to_sma = compute_price_to_sma_ratio(series, period=3)
    assert math.isclose(p_to_sma[2], (30.0 - 20.0) / 20.0)

    sma_spread = compute_sma_ratio(series, fast_period=2, slow_period=3)
    # At index 2: fast_sma=(20+30)/2=25, slow_sma=20 -> (25-20)/20 = 0.25
    assert math.isclose(sma_spread[2], 0.25)


# -------------------------------------------------------------------------
# 4. Volatility Feature Tests
# -------------------------------------------------------------------------


def test_volatility_features_math() -> None:
    """Test rolling std, realized volatility, True Range, and ATR."""
    returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    r_std = compute_rolling_std(returns, period=3)
    assert np.isnan(r_std[0])
    assert np.isnan(r_std[1])
    expected_sample_std = float(np.std([0.01, -0.02, 0.03], ddof=1))
    assert math.isclose(r_std[2], expected_sample_std, rel_tol=1e-5)

    close = np.array([100.0, 102.0, 101.0, 104.0, 103.0])
    high = np.array([101.0, 103.0, 102.5, 105.0, 104.0])
    low = np.array([99.0, 100.5, 100.0, 101.5, 102.0])

    tr = compute_true_range(high, low, close)
    assert len(tr) == len(close)
    assert math.isclose(tr[0], 2.0)  # 101 - 99

    atr = compute_average_true_range(high, low, close, period=2)
    assert np.isnan(atr[0])
    assert math.isclose(atr[1], (tr[0] + tr[1]) / 2.0)

    natr = compute_normalized_atr(high, low, close, period=2)
    assert math.isclose(natr[1], (atr[1] / close[1]) * 100.0)

    rv = compute_realized_volatility(close, period=3, periods_per_year=252)
    assert len(rv) == len(close)


# -------------------------------------------------------------------------
# 5. Volume Feature Tests
# -------------------------------------------------------------------------


def test_volume_features_math() -> None:
    """Test volume change, volume to SMA ratio, and volume Z-score."""
    vols = np.array([1000.0, 1200.0, 1500.0, 900.0, 1100.0])

    v_chg = compute_volume_change(vols, period=1)
    assert np.isnan(v_chg[0])
    assert math.isclose(v_chg[1], 0.20)  # (1200-1000)/1000

    v_sma = compute_volume_to_sma_ratio(vols, period=2)
    assert np.isnan(v_sma[0])
    # At index 1: sma = (1000+1200)/2 = 1100. 1200 / 1100 = 1.0909
    assert math.isclose(v_sma[1], 1200.0 / 1100.0)

    z = compute_volume_zscore(vols, period=3)
    assert np.isnan(z[0])
    assert np.isnan(z[1])
    w = vols[:3]
    expected_z = (vols[2] - np.mean(w)) / (np.std(w, ddof=1) + 1e-8)
    assert math.isclose(z[2], expected_z, rel_tol=1e-5)


# -------------------------------------------------------------------------
# 6. Price Structure Tests
# -------------------------------------------------------------------------


def test_price_structure_invariants() -> None:
    """Verify bounds of price structure features."""
    close = np.array([100.0, 105.0, 95.0, 102.0, 110.0])
    high = np.array([102.0, 107.0, 98.0, 104.0, 112.0])
    low = np.array([99.0, 103.0, 94.0, 99.0, 108.0])

    d_high = compute_distance_from_rolling_high(close, high, period=3)
    d_low = compute_distance_from_rolling_low(close, low, period=3)
    r_pos = compute_normalized_range_position(close, high, low, period=3)

    for i in range(2, len(close)):
        # Distance from rolling high must be non-positive
        assert d_high[i] <= 1e-8
        # Distance from rolling low must be non-negative
        assert d_low[i] >= -1e-8
        # Normalized position must be within [0.0, 1.0]
        assert 0.0 <= r_pos[i] <= 1.0


# -------------------------------------------------------------------------
# 7. CRITICAL: Look-Ahead Leakage Invariance Test
# -------------------------------------------------------------------------


def test_lookahead_leakage_invariance_across_all_registered_features() -> None:
    """Mutating future data points (t > T) MUST NEVER alter feature values at t <= T.

    This test mathematically proves the absence of future lookahead bias.
    """
    provider = SyntheticDataProvider(seed=123)
    records, _ = provider.generate_valid_dataset(symbol="TEST", num_bars=60)

    # Convert records to dictionary of numpy arrays
    ts = np.array([r["timestamp"] for r in records])
    o_arr = np.array([r["open"] for r in records], dtype=np.float64)
    h_arr = np.array([r["high"] for r in records], dtype=np.float64)
    l_arr = np.array([r["low"] for r in records], dtype=np.float64)
    c_arr = np.array([r["close"] for r in records], dtype=np.float64)
    v_arr = np.array([r["volume"] for r in records], dtype=np.float64)

    base_data = {
        "timestamp": ts,
        "open": o_arr.copy(),
        "high": h_arr.copy(),
        "low": l_arr.copy(),
        "close": c_arr.copy(),
        "volume": v_arr.copy(),
    }

    # Target timestamp index
    target_t = 35

    # Mutate FUTURE data (indices > target_t) violently
    mutated_data = {
        "timestamp": ts,
        "open": o_arr.copy(),
        "high": h_arr.copy(),
        "low": l_arr.copy(),
        "close": c_arr.copy(),
        "volume": v_arr.copy(),
    }
    mutated_data["open"][target_t + 1 :] *= 5.0
    mutated_data["high"][target_t + 1 :] *= 10.0
    mutated_data["low"][target_t + 1 :] *= 0.1
    mutated_data["close"][target_t + 1 :] *= 8.0
    mutated_data["volume"][target_t + 1 :] *= 100.0

    # Compute and verify EVERY registered feature
    for meta in registry.list_features():
        func = registry.get(meta.name)

        base_res = func(base_data)
        mut_res = func(mutated_data)

        # Slice up to and including target_t
        base_slice = base_res[: target_t + 1]
        mut_slice = mut_res[: target_t + 1]

        # For NaN values (warmup), ensure both are NaN
        nan_mask_base = np.isnan(base_slice)
        nan_mask_mut = np.isnan(mut_slice)
        assert np.array_equal(nan_mask_base, nan_mask_mut), (
            f"Feature {meta.name} changed NaN pattern at or before t={target_t}"
        )

        # For valid numerical values, assert identical down to 1e-12 precision
        valid = ~nan_mask_base
        diff = np.abs(base_slice[valid] - mut_slice[valid])
        max_diff = float(np.max(diff)) if len(diff) > 0 else 0.0

        err_msg = (
            f"LEAKAGE in '{meta.name}': mutation at t > {target_t} "
            f"altered historical values. diff={max_diff}"
        )
        assert max_diff < 1e-12, err_msg


# -------------------------------------------------------------------------
# 8. Multi-Symbol Partition Independence Test
# -------------------------------------------------------------------------


def test_multi_symbol_partition_independence() -> None:
    """Ensure multi-symbol datasets do not cross-contaminate rolling calculations."""
    provider = SyntheticDataProvider(seed=42)
    spy_records, _ = provider.generate_valid_dataset(symbol="SPY", num_bars=30)
    aapl_records, _ = provider.generate_valid_dataset(symbol="AAPL", num_bars=30)

    # Interleave or combine both symbols into single table
    from trady.data.normalizer import records_to_arrow_table

    combined_table = records_to_arrow_table(list(spy_records) + list(aapl_records))

    pipeline = FeaturePipeline(
        config=FeatureConfig(selected_features=("return_1d", "dist_high_10d"))
    )
    feature_table, report = pipeline.compute(combined_table)

    assert report.is_valid is True
    assert set(report.symbols) == {"SPY", "AAPL"}
    assert len(feature_table) == 60


# -------------------------------------------------------------------------
# 9. NaN Policy Tests
# -------------------------------------------------------------------------


def test_nan_policies_in_pipeline() -> None:
    """Test 'keep', 'drop', and 'error' nan_policies."""
    provider = SyntheticDataProvider(seed=99)
    records, _ = provider.generate_valid_dataset(symbol="SPY", num_bars=25)
    from trady.data.normalizer import records_to_arrow_table

    table = records_to_arrow_table(records)

    # 1. keep
    p_keep = FeaturePipeline(
        config=FeatureConfig(selected_features=("return_5d",), nan_policy="keep")
    )
    t_keep, rep_keep = p_keep.compute(table)
    assert len(t_keep) == 25
    assert rep_keep.nan_counts["return_5d"] == 5

    # 2. drop
    p_drop = FeaturePipeline(
        config=FeatureConfig(selected_features=("return_5d",), nan_policy="drop")
    )
    t_drop, _ = p_drop.compute(table)
    assert len(t_drop) == 20  # 25 - 5 warmup rows dropped

    # 3. error
    p_err = FeaturePipeline(
        config=FeatureConfig(selected_features=("return_5d",), nan_policy="error")
    )
    with pytest.raises(ValueError, match="under 'error' policy"):
        p_err.compute(table)


# -------------------------------------------------------------------------
# 10. End-to-End Pipeline & Storage Tests
# -------------------------------------------------------------------------


def test_feature_pipeline_save_and_provenance(tmp_path: Path) -> None:
    """Verify Parquet serialization and companion audit metadata."""
    provider = SyntheticDataProvider(seed=777)
    records, _ = provider.generate_valid_dataset(symbol="SPY", num_bars=40)
    from trady.data.normalizer import records_to_arrow_table

    input_table = records_to_arrow_table(records)
    out_parquet = tmp_path / "spy_features.parquet"

    pipeline = FeaturePipeline(
        config=FeatureConfig(
            enabled_groups=("PRICE", "VOLATILITY"),
            nan_policy="keep",
        )
    )
    feat_table, report = pipeline.compute(input_table)
    assert report.is_valid is True

    saved = pipeline.save(feat_table, out_parquet)
    assert saved.exists()

    meta_file = out_parquet.with_suffix(".parquet.meta.json")
    assert meta_file.exists()

    # Verify DuckDB can query the feature dataset
    import duckdb

    conn = duckdb.connect()
    parquet_sql = str(saved).replace("\\", "/")
    res = conn.execute(f"SELECT COUNT(*) FROM '{parquet_sql}'").fetchone()
    assert res is not None
    assert res[0] == 40
