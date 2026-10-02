"""Causal Feature Pipeline for TRADY.

Transforms validated OHLCV datasets into feature datasets without lookahead bias.
"""

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# Ensure default features are registered
import trady.features.momentum  # noqa: F401
import trady.features.price  # noqa: F401
import trady.features.structure  # noqa: F401
import trady.features.volatility  # noqa: F401
import trady.features.volume  # noqa: F401
from trady.data.storage import calculate_file_sha256
from trady.features.config import FeatureConfig
from trady.features.registry import FeatureMetadata, FeatureRegistry, registry
from trady.utils.logging import get_logger

logger = get_logger("features.pipeline")


@dataclass(frozen=True)
class FeatureValidationReport:
    """Audit report produced during feature calculation and validation."""

    is_valid: bool
    total_records: int
    feature_count: int
    symbols: tuple[str, ...]
    nan_counts: dict[str, int]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class FeatureDatasetMetadata:
    """Provenance and audit metadata for serialized feature datasets."""

    feature_names: tuple[str, ...]
    feature_count: int
    record_count: int
    symbols: tuple[str, ...]
    start_time: str | None
    end_time: str | None
    nan_policy: str
    nan_counts: dict[str, int]
    source_file: str | None
    source_checksum: str | None
    created_at: str
    features: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to JSON-serializable dictionary."""
        return asdict(self)


class FeaturePipeline:
    """Causal feature computation pipeline.

    Workflow:
        Processed OHLCV -> Feature Configuration -> Feature Calculations ->
        Feature Validation -> Feature Dataset (Parquet)
    """

    def __init__(
        self,
        config: FeatureConfig | None = None,
        feat_registry: FeatureRegistry | None = None,
    ) -> None:
        self.config = config or FeatureConfig()
        self.registry = feat_registry or registry

    def resolve_features(self) -> list[FeatureMetadata]:
        """Resolve which features to compute based on configuration."""
        if self.config.selected_features is not None:
            metas: list[FeatureMetadata] = []
            for name in self.config.selected_features:
                metas.append(self.registry.get_metadata(name))
            return metas

        metas = []
        for meta in self.registry.list_features():
            if meta.feature_group in self.config.enabled_groups:
                metas.append(meta)
        return metas

    def compute(
        self,
        data: pa.Table | str | Path,
    ) -> tuple[pa.Table, FeatureValidationReport]:
        """Compute all configured features causally on the input dataset.

        Args:
            data: PyArrow Table or file path to Parquet/JSON processed market data.

        Returns:
            Tuple of (computed feature PyArrow Table, validation audit report).
        """
        table = self._load_input_table(data)
        self._validate_input_schema(table)

        # Ensure table is sorted chronologically by (symbol, timestamp)
        table = self._sort_table(table)

        selected_metas = self.resolve_features()
        if not selected_metas:
            raise ValueError("No features selected for computation")

        logger.info(
            "Starting feature pipeline: %d features, nan_policy=%s",
            len(selected_metas),
            self.config.nan_policy,
        )

        # Partition by symbol to prevent any cross-symbol leakage
        symbols_col = table.column("symbol").to_pylist()
        unique_symbols = sorted(set(symbols_col))

        out_timestamps: list[Any] = []
        out_symbols: list[str] = []
        feature_arrays: dict[str, list[np.ndarray]] = {
            m.name: [] for m in selected_metas
        }

        for symbol in unique_symbols:
            # Filter table for this symbol
            sym_table = table.filter(pc.equal(table.column("symbol"), symbol))

            # Extract arrays
            ts = sym_table.column("timestamp").to_numpy(zero_copy_only=False)
            open_arr = sym_table.column("open").to_numpy(zero_copy_only=False)
            high_arr = sym_table.column("high").to_numpy(zero_copy_only=False)
            low_arr = sym_table.column("low").to_numpy(zero_copy_only=False)
            close_arr = sym_table.column("close").to_numpy(zero_copy_only=False)
            vol_arr = sym_table.column("volume").to_numpy(zero_copy_only=False)

            symbol_data = {
                "timestamp": ts,
                "open": open_arr,
                "high": high_arr,
                "low": low_arr,
                "close": close_arr,
                "volume": vol_arr,
            }

            out_timestamps.extend(ts)
            out_symbols.extend([symbol] * len(ts))

            # Calculate each feature for this symbol partition
            for meta in selected_metas:
                func = self.registry.get(meta.name)
                # Verify required columns exist
                for req_col in meta.required_columns:
                    if req_col not in symbol_data:
                        msg = f"Feature '{meta.name}' missing column '{req_col}'"
                        raise ValueError(msg)

                feat_vals = func(symbol_data)
                feat_arr = np.asarray(feat_vals, dtype=np.float64)
                if len(feat_arr) != len(ts):
                    msg = f"Feature '{meta.name}' produced {len(feat_arr)} items"
                    raise RuntimeError(msg)
                feature_arrays[meta.name].append(feat_arr)

        # Concatenate symbol partitions
        concatenated_features: dict[str, np.ndarray] = {}
        nan_counts: dict[str, int] = {}
        for meta in selected_metas:
            concatenated = np.concatenate(feature_arrays[meta.name])
            concatenated_features[meta.name] = concatenated
            nan_counts[meta.name] = int(np.isnan(concatenated).sum())

        total_rows = len(out_timestamps)

        # Apply NaN Policy
        valid_indices = np.ones(total_rows, dtype=bool)
        if self.config.nan_policy == "drop":
            for feat_arr in concatenated_features.values():
                valid_indices &= ~np.isnan(feat_arr)
            logger.info(
                "Applied 'drop' NaN policy: dropped %d rows, retained %d rows",
                int((~valid_indices).sum()),
                int(valid_indices.sum()),
            )
        elif self.config.nan_policy == "error":
            for name, count in nan_counts.items():
                if count > 0:
                    msg = f"Feature '{name}' has {count} NaNs under 'error' policy"
                    raise ValueError(msg)

        # Construct final arrays
        filtered_timestamps = np.asarray(out_timestamps)[valid_indices]
        filtered_symbols = [
            out_symbols[i] for i in range(total_rows) if valid_indices[i]
        ]

        # Build PyArrow schema and table
        arrow_arrays: list[pa.Array] = [
            pa.array(filtered_timestamps),
            pa.array(filtered_symbols, type=pa.string()),
        ]
        arrow_names: list[str] = ["timestamp", "symbol"]

        for meta in selected_metas:
            filtered_feat = concatenated_features[meta.name][valid_indices]
            arrow_arrays.append(pa.array(filtered_feat, type=pa.float64()))
            arrow_names.append(meta.name)

        feature_table = pa.Table.from_arrays(arrow_arrays, names=arrow_names)

        # Audit and Validate
        errors: list[str] = []
        warnings: list[str] = []

        if len(feature_table) == 0:
            warnings.append("Feature table is empty after computation / NaN filtering")

        # Check for infinite values
        for meta in selected_metas:
            col_arr = concatenated_features[meta.name][valid_indices]
            if np.isinf(col_arr).any():
                errors.append(f"Feature '{meta.name}' contains infinite (Inf) values")

        report = FeatureValidationReport(
            is_valid=len(errors) == 0,
            total_records=len(feature_table),
            feature_count=len(selected_metas),
            symbols=tuple(unique_symbols),
            nan_counts=nan_counts,
            errors=tuple(errors),
            warnings=tuple(warnings),
        )

        return feature_table, report

    def save(
        self,
        table: pa.Table,
        output_path: str | Path,
        source_path: str | Path | None = None,
    ) -> Path:
        """Serialize feature table to Parquet and persist companion .meta.json.

        Args:
            table: PyArrow Table of computed features.
            output_path: Target path for the Parquet dataset.
            source_path: Optional path to the source market data file for provenance.

        Returns:
            Resolved Path to the saved Parquet file.
        """
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        pq.write_table(table, out_p, compression="snappy")

        # Compute checksum of written parquet file
        checksum = calculate_file_sha256(out_p)

        # Collect symbols and timestamps
        symbols = (
            sorted(set(table.column("symbol").to_pylist()))
            if "symbol" in table.column_names
            else []
        )
        start_time_str = None
        end_time_str = None
        if "timestamp" in table.column_names and len(table) > 0:
            ts_col = table.column("timestamp").to_pylist()
            start_time_str = (
                ts_col[0].isoformat()
                if hasattr(ts_col[0], "isoformat")
                else str(ts_col[0])
            )
            end_time_str = (
                ts_col[-1].isoformat()
                if hasattr(ts_col[-1], "isoformat")
                else str(ts_col[-1])
            )

        feature_cols = [
            c for c in table.column_names if c not in ("timestamp", "symbol")
        ]

        # Collect metadata for each feature
        features_info: list[dict[str, Any]] = []
        for name in feature_cols:
            if self.registry.contains(name):
                m = self.registry.get_metadata(name)
                features_info.append(
                    {
                        "name": m.name,
                        "description": m.description,
                        "group": m.feature_group,
                        "lookback_period": m.lookback_period,
                        "required_columns": list(m.required_columns),
                        "is_point_in_time": m.is_point_in_time,
                    }
                )
            else:
                features_info.append({"name": name, "custom": True})

        nan_counts = {col: int(table.column(col).null_count) for col in feature_cols}

        source_checksum = None
        if source_path is not None and Path(source_path).exists():
            source_checksum = calculate_file_sha256(Path(source_path))

        meta = FeatureDatasetMetadata(
            feature_names=tuple(feature_cols),
            feature_count=len(feature_cols),
            record_count=len(table),
            symbols=tuple(symbols),
            start_time=start_time_str,
            end_time=end_time_str,
            nan_policy=self.config.nan_policy,
            nan_counts=nan_counts,
            source_file=str(source_path) if source_path else None,
            source_checksum=source_checksum,
            created_at=datetime.now(UTC).isoformat(),
            features=features_info,
        )

        meta_file = out_p.with_suffix(out_p.suffix + ".meta.json")
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta.to_dict(), f, indent=4)

        logger.info(
            "Feature dataset persisted: %s (%d records, sha256=%s)",
            out_p,
            len(table),
            checksum,
        )
        return out_p

    def _load_input_table(self, data: pa.Table | str | Path) -> pa.Table:
        """Load input data into PyArrow Table."""
        if isinstance(data, pa.Table):
            return data

        path = Path(data)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        if path.suffix in (".parquet", ".pq"):
            return pq.read_table(path)
        elif path.suffix in (".json",):
            from trady.data.normalizer import normalize_records, records_to_arrow_table

            with open(path, encoding="utf-8") as f:
                raw_data = json.load(f)
            records = normalize_records(raw_data)
            return records_to_arrow_table(records)
        else:
            raise ValueError(
                f"Unsupported file format '{path.suffix}'. Use Parquet or JSON."
            )

    def _validate_input_schema(self, table: pa.Table) -> None:
        """Verify the input table has canonical OHLCV columns."""
        required = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
        missing = required - set(table.column_names)
        if missing:
            raise ValueError(
                f"Input market data missing required canonical columns: {missing}"
            )

    def _sort_table(self, table: pa.Table) -> pa.Table:
        """Sort table chronologically by symbol and timestamp ascending."""
        return table.sort_by([("symbol", "ascending"), ("timestamp", "ascending")])
