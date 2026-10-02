"""ModelingDataset container enforcing strict feature-target separation.

Prevents target leakage by design:
- Features (X): information available at prediction time t <= T.
- Targets (y): future outcomes being predicted t > T.
- Invariant: feature_names and target_names must be strictly disjoint sets.
"""

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from trady.data.storage import calculate_file_sha256
from trady.targets.metadata import TargetMetadata


@dataclass(frozen=True)
class ModelingDatasetMetadata:
    """Provenance and schema metadata for modeling datasets."""

    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    record_count: int
    symbols: tuple[str, ...]
    start_time: str | None
    end_time: str | None
    targets_info: list[dict[str, Any]]
    source_checksum: str | None
    created_at: str
    dataset_checksum: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
        return asdict(self)


class ModelingDataset:
    """Container for paired features (X) and forward targets (y).

    Guarantees that targets can never be accidentally accessed as features.
    """

    def __init__(
        self,
        table: pa.Table,
        feature_names: tuple[str, ...],
        target_names: tuple[str, ...],
        target_metadatas: dict[str, TargetMetadata] | None = None,
        timestamp_col: str = "timestamp",
        symbol_col: str = "symbol",
    ) -> None:
        self.table = table
        self.feature_names = tuple(feature_names)
        self.target_names = tuple(target_names)
        self.target_metadatas = target_metadatas or {}
        self.timestamp_col = timestamp_col
        self.symbol_col = symbol_col

        self._validate_leakage_invariants()

    def __len__(self) -> int:
        """Return total number of rows in the dataset."""
        return len(self.table)

    def _validate_leakage_invariants(self) -> None:
        """Enforce strict disjointness between features and targets."""
        # 1. Check disjointness
        overlap = set(self.feature_names).intersection(set(self.target_names))
        if overlap:
            msg = (
                "TARGET LEAKAGE DETECTED! Columns "
                f"{overlap} appear in both features and targets."
            )
            raise ValueError(msg)

        # 2. Check presence in table
        all_cols = set(self.table.column_names)
        missing_feats = set(self.feature_names) - all_cols
        if missing_feats:
            msg = f"Feature columns missing from table: {missing_feats}"
            raise ValueError(msg)

        missing_targets = set(self.target_names) - all_cols
        if missing_targets:
            msg = f"Target columns missing from table: {missing_targets}"
            raise ValueError(msg)

    def get_features(self, drop_na: bool = False) -> np.ndarray:
        """Extract feature matrix X.

        Targets are NEVER included in the returned matrix.

        Args:
            drop_na: Whether to filter out rows containing NaNs.

        Returns:
            2D numpy float64 array of shape (N, num_features).
        """
        cols = [
            self.table.column(f).to_numpy(zero_copy_only=False)
            for f in self.feature_names
        ]
        x_mat = np.column_stack(cols).astype(np.float64)

        if drop_na:
            valid_mask = ~np.isnan(x_mat).any(axis=1)
            return x_mat[valid_mask]
        return x_mat

    # Convenience alias matching mathematical notation
    get_x = get_features

    def get_target(self, target_name: str, drop_na: bool = False) -> np.ndarray:
        """Extract target vector y for a specific target.

        Args:
            target_name: Name of target to extract.
            drop_na: Whether to filter out rows containing NaN.

        Returns:
            1D numpy float64 array of target values.
        """
        if target_name not in self.target_names:
            msg = (
                f"Target '{target_name}' not in dataset. "
                f"Available targets: {self.target_names}"
            )
            raise KeyError(msg)

        y_vec = (
            self.table.column(target_name)
            .to_numpy(zero_copy_only=False)
            .astype(np.float64)
        )
        if drop_na:
            valid_mask = ~np.isnan(y_vec)
            return y_vec[valid_mask]
        return y_vec

    # Convenience alias matching mathematical notation
    get_y = get_target

    def get_modeling_arrays(
        self,
        target_name: str,
        drop_na: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Extract aligned (X, y, timestamps, symbols) for model training.

        Automatically prunes warmup NaNs (from features) and trailing NaNs
        (from targets).

        Args:
            target_name: Name of the target variable to predict.
            drop_na: If True, only rows where BOTH X and y are valid are kept.

        Returns:
            Tuple of (X 2D array, y 1D array, timestamps 1D array, symbols list).
        """
        if target_name not in self.target_names:
            msg = (
                f"Target '{target_name}' not found. "
                f"Available targets: {self.target_names}"
            )
            raise KeyError(msg)

        cols = [
            self.table.column(f).to_numpy(zero_copy_only=False)
            for f in self.feature_names
        ]
        x_mat = np.column_stack(cols).astype(np.float64)
        y_vec = (
            self.table.column(target_name)
            .to_numpy(zero_copy_only=False)
            .astype(np.float64)
        )
        ts_arr = self.table.column(self.timestamp_col).to_numpy(zero_copy_only=False)
        symbols = self.table.column(self.symbol_col).to_pylist()

        if drop_na:
            valid_x = ~np.isnan(x_mat).any(axis=1)
            valid_y = ~np.isnan(y_vec)
            valid = valid_x & valid_y

            x_mat = x_mat[valid]
            y_vec = y_vec[valid]
            ts_arr = ts_arr[valid]
            symbols = [symbols[i] for i in range(len(symbols)) if valid[i]]

        return x_mat, y_vec, ts_arr, symbols

    def save(
        self,
        output_path: str | Path,
        source_path: str | Path | None = None,
    ) -> Path:
        """Serialize dataset to Parquet with companion metadata.

        Args:
            output_path: Path to write Parquet file.
            source_path: Optional source data file path for checksum audit.

        Returns:
            Resolved Path to written Parquet file.
        """
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        pq.write_table(self.table, out_p, compression="snappy")

        checksum = calculate_file_sha256(out_p)
        source_checksum = (
            calculate_file_sha256(Path(source_path))
            if source_path and Path(source_path).exists()
            else None
        )

        symbols = (
            sorted(set(self.table.column(self.symbol_col).to_pylist()))
            if self.symbol_col in self.table.column_names
            else []
        )
        start_time_str = None
        end_time_str = None
        if self.timestamp_col in self.table.column_names and len(self.table) > 0:
            ts_col = self.table.column(self.timestamp_col).to_pylist()
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

        targets_info = []
        for t_name in self.target_names:
            if t_name in self.target_metadatas:
                m = self.target_metadatas[t_name]
                targets_info.append(
                    {
                        "name": m.name,
                        "horizon": m.horizon,
                        "target_type": m.target_type,
                        "threshold": m.threshold,
                        "calculation_method": m.calculation_method,
                        "description": m.description,
                    }
                )
            else:
                targets_info.append({"name": t_name, "custom": True})

        meta = ModelingDatasetMetadata(
            feature_names=self.feature_names,
            target_names=self.target_names,
            record_count=len(self.table),
            symbols=tuple(symbols),
            start_time=start_time_str,
            end_time=end_time_str,
            targets_info=targets_info,
            source_checksum=source_checksum,
            created_at=datetime.now(UTC).isoformat(),
            dataset_checksum=checksum,
        )

        meta_file = out_p.with_suffix(out_p.suffix + ".meta.json")
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta.to_dict(), f, indent=4)

        return out_p

    @classmethod
    def load(cls, path: str | Path) -> "ModelingDataset":
        """Load a persisted modeling dataset from Parquet and metadata.

        Args:
            path: Path to the Parquet file.

        Returns:
            Instantiated ModelingDataset with separated features and targets.
        """
        parquet_path = Path(path)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {parquet_path}")

        table = pq.read_table(parquet_path)
        meta_file = parquet_path.with_suffix(parquet_path.suffix + ".meta.json")

        if meta_file.exists():
            with open(meta_file, encoding="utf-8") as f:
                meta = json.load(f)
            feature_names = tuple(meta.get("feature_names", ()))
            target_names = tuple(meta.get("target_names", ()))
        else:
            # Fallback: infer from column naming conventions
            target_names = tuple(
                c for c in table.column_names if c.startswith("target_")
            )
            ignored = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
            feature_names = tuple(
                c
                for c in table.column_names
                if c not in ignored and not c.startswith("target_")
            )

        return cls(
            table=table,
            feature_names=feature_names,
            target_names=target_names,
        )
