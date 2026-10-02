"""Target computation pipeline for TRADY quantitative experiments."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from trady.targets.builder import register_default_targets
from trady.targets.config import TargetConfig
from trady.targets.dataset import ModelingDataset
from trady.targets.metadata import TargetMetadata
from trady.targets.registry import TargetRegistry, target_registry
from trady.utils.logging import get_logger

logger = get_logger("targets.pipeline")

# Ensure default targets are registered
register_default_targets()


@dataclass(frozen=True)
class TargetReport:
    """Audit report for computed targets."""

    total_records: int
    target_names: tuple[str, ...]
    symbols: tuple[str, ...]
    unavailable_counts: dict[str, int]
    is_valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


class TargetPipeline:
    """Pipeline for computing forward-looking targets and modeling datasets."""

    def __init__(
        self,
        config: TargetConfig | None = None,
        registry: TargetRegistry | None = None,
    ) -> None:
        self.config = config or TargetConfig()
        self.registry = registry or target_registry

    def resolve_targets(self) -> list[TargetMetadata]:
        """Resolve metadata for configured targets."""
        metas: list[TargetMetadata] = []
        for name in self.config.selected_targets:
            metas.append(self.registry.get_metadata(name))
        return metas

    def compute(
        self,
        market_data: pa.Table | str | Path,
        features_data: pa.Table | str | Path | None = None,
    ) -> tuple[ModelingDataset, TargetReport]:
        """Compute targets partitioned by symbol and construct ModelingDataset.

        Args:
            market_data: Processed OHLCV table or file path containing price data.
            features_data: Optional computed feature table or file path.

        Returns:
            Tuple of (ModelingDataset, TargetReport).
        """
        table = self._load_table(market_data)
        self._validate_market_data(table)

        # Sort chronologically by symbol and timestamp
        table = table.sort_by([("symbol", "ascending"), ("timestamp", "ascending")])

        # If features_data is provided, join or merge with market_data
        feat_table: pa.Table | None = None
        if features_data is not None:
            feat_table = self._load_table(features_data)
            table = self._merge_market_and_features(table, feat_table)

        selected_metas = self.resolve_targets()
        logger.info(
            "Starting target pipeline: %d targets, drop_unavailable=%s",
            len(selected_metas),
            self.config.drop_unavailable_rows,
        )

        symbols_col = table.column("symbol").to_pylist()
        unique_symbols = sorted(set(symbols_col))

        out_timestamps: list[Any] = []
        out_symbols: list[str] = []
        target_arrays: dict[str, list[np.ndarray]] = {
            m.name: [] for m in selected_metas
        }

        # Partition by symbol to prevent any cross-symbol lookahead
        for symbol in unique_symbols:
            sym_table = table.filter(pc.equal(table.column("symbol"), symbol))

            ts = sym_table.column("timestamp").to_numpy(zero_copy_only=False)
            close_arr = sym_table.column("close").to_numpy(zero_copy_only=False)
            high_arr = sym_table.column("high").to_numpy(zero_copy_only=False)
            low_arr = sym_table.column("low").to_numpy(zero_copy_only=False)

            symbol_data = {
                "timestamp": ts,
                "close": close_arr,
                "high": high_arr,
                "low": low_arr,
            }

            out_timestamps.extend(ts)
            out_symbols.extend([symbol] * len(ts))

            for meta in selected_metas:
                func = self.registry.get(meta.name)
                tgt_vals = func(symbol_data)
                tgt_arr = np.asarray(tgt_vals, dtype=np.float64)
                if len(tgt_arr) != len(ts):
                    msg = f"Target '{meta.name}' produced {len(tgt_arr)} items"
                    raise RuntimeError(msg)
                target_arrays[meta.name].append(tgt_arr)

        # Concatenate symbol partitions
        concatenated_targets: dict[str, np.ndarray] = {}
        unavailable_counts: dict[str, int] = {}
        for meta in selected_metas:
            concatenated = np.concatenate(target_arrays[meta.name])
            concatenated_targets[meta.name] = concatenated
            unavailable_counts[meta.name] = int(np.isnan(concatenated).sum())

        total_rows = len(table)
        valid_indices = np.ones(total_rows, dtype=bool)

        if self.config.drop_unavailable_rows:
            for tgt_arr in concatenated_targets.values():
                valid_indices &= ~np.isnan(tgt_arr)
            logger.info(
                "Dropped %d unavailable trailing target rows",
                int((~valid_indices).sum()),
            )

        # Build final table
        new_arrays = [col for col in table]
        new_names = list(table.column_names)

        # Append computed target columns
        target_names = [m.name for m in selected_metas]
        for meta in selected_metas:
            tgt_arr = concatenated_targets[meta.name]
            new_arrays.append(pa.array(tgt_arr, type=pa.float64()))
            new_names.append(meta.name)

        combined_table = pa.Table.from_arrays(new_arrays, names=new_names)

        if self.config.drop_unavailable_rows:
            combined_table = combined_table.filter(pa.array(valid_indices))

        # Identify feature columns (excluding identifiers, raw OHLCV, and targets)
        ignored = {
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
        }.union(target_names)
        feature_cols = [c for c in combined_table.column_names if c not in ignored]

        # Map target metadata
        target_meta_map = {m.name: m for m in selected_metas}

        modeling_dataset = ModelingDataset(
            table=combined_table,
            feature_names=tuple(feature_cols),
            target_names=tuple(target_names),
            target_metadatas=target_meta_map,
        )

        report = TargetReport(
            total_records=len(combined_table),
            target_names=tuple(target_names),
            symbols=tuple(unique_symbols),
            unavailable_counts=unavailable_counts,
            is_valid=True,
            errors=(),
            warnings=(),
        )

        return modeling_dataset, report

    def _load_table(self, data: pa.Table | str | Path) -> pa.Table:
        """Load data into a PyArrow Table."""
        if isinstance(data, pa.Table):
            return data

        path = Path(data)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path.suffix in (".parquet", ".pq"):
            return pq.read_table(path)
        else:
            raise ValueError(f"Unsupported file format '{path.suffix}'. Use Parquet.")

    def _validate_market_data(self, table: pa.Table) -> None:
        """Validate required market data columns."""
        required = {"timestamp", "symbol", "close"}
        missing = required - set(table.column_names)
        if missing:
            raise ValueError(
                f"Input data missing required columns for targets: {missing}"
            )

    def _merge_market_and_features(
        self,
        market_table: pa.Table,
        feature_table: pa.Table,
    ) -> pa.Table:
        """Merge market data table with feature table on (timestamp, symbol)."""
        # Ensure row counts and keys match
        if len(market_table) != len(feature_table):
            # Sort both by symbol and timestamp
            market_table = market_table.sort_by(
                [("symbol", "ascending"), ("timestamp", "ascending")]
            )
            feature_table = feature_table.sort_by(
                [("symbol", "ascending"), ("timestamp", "ascending")]
            )

        # Add feature columns not already in market_table
        new_arrays = [col for col in market_table]
        new_names = list(market_table.column_names)

        for name in feature_table.column_names:
            if name not in ("timestamp", "symbol") and name not in new_names:
                new_arrays.append(feature_table.column(name))
                new_names.append(name)

        return pa.Table.from_arrays(new_arrays, names=new_names)
