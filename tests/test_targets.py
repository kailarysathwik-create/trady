"""Unit and leakage-prevention tests for the TRADY Target Engine."""

import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from trady.data.provider import SyntheticDataProvider
from trady.targets.builder import (
    compute_binary_classification_target,
    compute_forward_return,
    compute_forward_volatility,
)
from trady.targets.config import TargetConfig
from trady.targets.dataset import ModelingDataset
from trady.targets.metadata import TargetMetadata
from trady.targets.pipeline import TargetPipeline
from trady.targets.registry import target_registry

# -------------------------------------------------------------------------
# 1. Target Metadata & Registry Tests
# -------------------------------------------------------------------------


def test_target_metadata_validation() -> None:
    """Verify TargetMetadata enforces valid horizons and types."""
    meta = TargetMetadata(
        name="target_fwd_ret_5d",
        horizon=5,
        target_type="continuous",
        calculation_method="forward_simple_return",
        description="Forward 5-day return",
    )
    assert meta.horizon == 5
    assert meta.is_future_target is True

    with pytest.raises(ValueError, match="at least 1 bar"):
        TargetMetadata(
            name="INVALID",
            horizon=0,
            target_type="continuous",
            calculation_method="test",
            description="test",
        )

    with pytest.raises(ValueError, match="Target name cannot be empty"):
        TargetMetadata(
            name="",
            horizon=5,
            target_type="continuous",
            calculation_method="test",
            description="test",
        )


def test_target_registry_contains_defaults() -> None:
    """Ensure default targets are loaded in the registry."""
    targets = {m.name for m in target_registry.list_targets()}
    assert "target_fwd_ret_1d" in targets
    assert "target_fwd_ret_5d" in targets
    assert "target_binary_up_5d" in targets
    assert "target_fwd_vol_5d" in targets


# -------------------------------------------------------------------------
# 2. Forward Return Math & Exact Alignment Tests
# -------------------------------------------------------------------------


def test_forward_return_exact_alignment_and_trailing_nans() -> None:
    """Prove that forward returns align to t and the last H bars are strictly NaN."""
    prices = np.array([100.0, 110.0, 121.0, 108.9])

    # Horizon H = 1
    fwd_1 = compute_forward_return(prices, horizon=1, method="simple")
    assert math.isclose(fwd_1[0], 0.10, rel_tol=1e-5)  # (110 - 100) / 100
    assert math.isclose(fwd_1[1], 0.10, rel_tol=1e-5)  # (121 - 110) / 110
    assert math.isclose(fwd_1[2], -0.10, rel_tol=1e-5)  # (108.9 - 121) / 121
    assert np.isnan(fwd_1[3])  # Last bar has no future price

    # Horizon H = 2
    fwd_2 = compute_forward_return(prices, horizon=2, method="simple")
    assert math.isclose(fwd_2[0], 0.21, rel_tol=1e-5)  # (121 - 100) / 100
    assert math.isclose(fwd_2[1], (108.9 - 110.0) / 110.0, rel_tol=1e-5)
    assert np.isnan(fwd_2[2])  # Last 2 bars must be NaN
    assert np.isnan(fwd_2[3])

    # Forward Log Return
    log_fwd_1 = compute_forward_return(prices, horizon=1, method="log")
    assert math.isclose(log_fwd_1[0], math.log(1.10), rel_tol=1e-5)
    assert np.isnan(log_fwd_1[3])


# -------------------------------------------------------------------------
# 3. Binary Classification Target Tests
# -------------------------------------------------------------------------


def test_binary_classification_target_and_threshold() -> None:
    """Test binary classification labels and ensure trailing rows are NaN, not 0."""
    prices = np.array([100.0, 105.0, 95.0, 110.0])

    # Directional up (> 0.0)
    b_up = compute_binary_classification_target(prices, horizon=1, threshold=0.0)
    assert b_up[0] == 1.0  # 105 > 100 (+5%)
    assert b_up[1] == 0.0  # 95 < 105 (-9.5%)
    assert b_up[2] == 1.0  # 110 > 95 (+15.8%)
    assert np.isnan(b_up[3])  # CRITICAL: Future is unknown, must NOT be 0.0!

    # Threshold > 0.10 (+10%)
    b_thresh = compute_binary_classification_target(prices, horizon=1, threshold=0.10)
    assert b_thresh[0] == 0.0  # +5% <= +10%
    assert b_thresh[1] == 0.0  # -9.5% <= +10%
    assert b_thresh[2] == 1.0  # +15.8% > +10%
    assert np.isnan(b_thresh[3])


# -------------------------------------------------------------------------
# 4. Forward Volatility Target Tests
# -------------------------------------------------------------------------


def test_forward_volatility_math_and_trailing_nans() -> None:
    """Test forward realized volatility over future H bars."""
    prices = np.array([100.0, 102.0, 101.0, 105.0, 104.0])
    h = 2
    fwd_vol = compute_forward_volatility(prices, horizon=h, annualize=False)

    # For t=0: future 1-day returns are r1 (102 vs 100 = 0.02)
    # and r2 (101 vs 102 = -0.0098)
    expected_std_0 = float(np.std([0.02, (101.0 - 102.0) / 102.0], ddof=1))
    assert math.isclose(fwd_vol[0], expected_std_0, rel_tol=1e-5)

    # The last 2 bars must be NaN
    assert np.isnan(fwd_vol[3])
    assert np.isnan(fwd_vol[4])


# -------------------------------------------------------------------------
# 5. CRITICAL: Feature-Target Disjointness & Anti-Leakage Tests
# -------------------------------------------------------------------------


def test_modeling_dataset_enforces_strict_disjointness() -> None:
    """ModelingDataset MUST reject columns in both features and targets."""
    table = pa.table(
        {
            "timestamp": [1, 2, 3],
            "symbol": ["A", "A", "A"],
            "return_1d": [0.01, 0.02, 0.03],
            "target_fwd_ret_1d": [0.02, 0.03, np.nan],
        }
    )

    # Valid separation
    ds = ModelingDataset(
        table=table,
        feature_names=("return_1d",),
        target_names=("target_fwd_ret_1d",),
    )
    assert ds.feature_names == ("return_1d",)
    assert ds.target_names == ("target_fwd_ret_1d",)

    # Invalid: Overlap / Target Leakage
    with pytest.raises(ValueError, match="TARGET LEAKAGE DETECTED"):
        ModelingDataset(
            table=table,
            feature_names=("return_1d", "target_fwd_ret_1d"),
            target_names=("target_fwd_ret_1d",),
        )


def test_feature_matrix_never_contains_target_columns() -> None:
    """get_features() must NEVER include target data in the extracted matrix."""
    table = pa.table(
        {
            "timestamp": [1, 2, 3],
            "symbol": ["A", "A", "A"],
            "feat_a": [1.0, 2.0, 3.0],
            "feat_b": [4.0, 5.0, 6.0],
            "target_fwd_ret_5d": [0.05, -0.02, np.nan],
            "target_binary_up_5d": [1.0, 0.0, np.nan],
        }
    )

    ds = ModelingDataset(
        table=table,
        feature_names=("feat_a", "feat_b"),
        target_names=("target_fwd_ret_5d", "target_binary_up_5d"),
    )

    features_mat = ds.get_features()
    assert features_mat.shape == (3, 2)
    # Ensure values match feat_a and feat_b only
    np.testing.assert_array_equal(features_mat[:, 0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(features_mat[:, 1], [4.0, 5.0, 6.0])

    # Test alias get_x()
    np.testing.assert_array_equal(ds.get_x(), features_mat)

    y = ds.get_y("target_fwd_ret_5d")
    assert len(y) == 3
    assert y[0] == 0.05
    assert np.isnan(y[2])


def test_modeling_arrays_alignment_prunes_warmup_and_trailing() -> None:
    """get_modeling_arrays(drop_na=True) drops both warmup and trailing NaN rows."""
    table = pa.table(
        {
            "timestamp": [1, 2, 3, 4, 5],
            "symbol": ["A", "A", "A", "A", "A"],
            "feature_warmup": [np.nan, np.nan, 3.0, 4.0, 5.0],  # warmup NaNs at 0, 1
            "target_trailing": [0.1, 0.2, 0.3, np.nan, np.nan],  # trailing NaNs at 3, 4
        }
    )

    ds = ModelingDataset(
        table=table,
        feature_names=("feature_warmup",),
        target_names=("target_trailing",),
    )

    x_mat, y, ts, syms = ds.get_modeling_arrays("target_trailing", drop_na=True)
    # Only row index 2 has BOTH valid feature and valid target!
    assert len(x_mat) == 1
    assert len(y) == 1
    assert x_mat[0, 0] == 3.0
    assert y[0] == 0.3
    assert ts[0] == 3
    assert syms == ["A"]


# -------------------------------------------------------------------------
# 6. Multi-Symbol Target Independence Test
# -------------------------------------------------------------------------


def test_multi_symbol_target_independence() -> None:
    """Ensure future targets do not look across symbol boundaries."""
    provider = SyntheticDataProvider(seed=55)
    spy_records, _ = provider.generate_valid_dataset(symbol="SPY", num_bars=20)
    aapl_records, _ = provider.generate_valid_dataset(symbol="AAPL", num_bars=20)

    from trady.data.normalizer import records_to_arrow_table

    combined = records_to_arrow_table(list(spy_records) + list(aapl_records))

    pipeline = TargetPipeline(
        config=TargetConfig(selected_targets=("target_fwd_ret_5d",))
    )
    ds, report = pipeline.compute(combined)

    assert report.is_valid is True
    # The last 5 bars of SPY must be NaN, and the last 5 bars of AAPL must be NaN
    # SPY is rows 0..19, AAPL is rows 20..39
    y = ds.get_y("target_fwd_ret_5d")

    # SPY trailing 5 bars (indices 15..19) must be NaN
    assert np.isnan(y[15:20]).all(), "SPY trailing rows leaked into AAPL!"
    # AAPL trailing 5 bars (indices 35..39) must be NaN
    assert np.isnan(y[35:40]).all()


# -------------------------------------------------------------------------
# 7. Pipeline Roundtrip & Storage Tests
# -------------------------------------------------------------------------


def test_target_pipeline_save_and_provenance(tmp_path: Path) -> None:
    """Verify saving ModelingDataset to Parquet and metadata inspection."""
    provider = SyntheticDataProvider(seed=88)
    records, _ = provider.generate_valid_dataset(symbol="SPY", num_bars=30)
    from trady.data.normalizer import records_to_arrow_table

    market_table = records_to_arrow_table(records)

    pipeline = TargetPipeline(
        config=TargetConfig(
            selected_targets=("target_fwd_ret_1d", "target_binary_up_1d"),
            drop_unavailable_rows=False,
        )
    )
    dataset, report = pipeline.compute(market_table)
    assert report.is_valid is True

    out_file = tmp_path / "labeled_spy.parquet"
    saved = dataset.save(out_file)
    assert saved.exists()

    meta_file = out_file.with_suffix(".parquet.meta.json")
    assert meta_file.exists()

    # Verify ModelingDataset.load
    loaded = ModelingDataset.load(saved)
    assert loaded.target_names == dataset.target_names
    assert len(loaded) == len(dataset)
    assert loaded.feature_names == dataset.feature_names
