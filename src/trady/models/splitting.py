"""Chronological time-series splitting for quantitative research.

CRITICAL FINANCIAL TIME-SERIES INVARIANTS:
1. Strictly chronological: train < validation < test.
2. NO random train/test shuffling: destroys autocorrelation and causes leakage.
3. Purge gap / Embargo: optional gap of H bars before test/validation splits
   to prevent forward-looking target horizons from bleeding into evaluation.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ChronologicalSplit:
    """Container holding index partitions and temporal boundaries."""

    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    train_start: str
    train_end: str
    val_start: str | None
    val_end: str | None
    test_start: str
    test_end: str
    purge_gap: int

    @property
    def train_count(self) -> int:
        """Number of training samples."""
        return len(self.train_indices)

    @property
    def val_count(self) -> int:
        """Number of validation samples."""
        return len(self.val_indices)

    @property
    def test_count(self) -> int:
        """Number of test samples."""
        return len(self.test_indices)

    def to_dict(self) -> dict[str, Any]:
        """Convert split summary to dictionary."""
        return {
            "train": {
                "count": self.train_count,
                "start": self.train_start,
                "end": self.train_end,
            },
            "validation": {
                "count": self.val_count,
                "start": self.val_start,
                "end": self.val_end,
            },
            "test": {
                "count": self.test_count,
                "start": self.test_start,
                "end": self.test_end,
            },
            "purge_gap": self.purge_gap,
        }


class ChronologicalSplitter:
    """Splits time-series data chronologically without shuffling."""

    @staticmethod
    def split_by_ratio(
        timestamps: np.ndarray | list[Any],
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        purge_gap: int = 0,
    ) -> ChronologicalSplit:
        """Split ordered data by ratio strictly preserving temporal order.

        Args:
            timestamps: 1D array of timestamps corresponding to data rows.
            train_ratio: Fraction of data for training (e.g. 0.6).
            val_ratio: Fraction of data for validation (e.g. 0.2).
            test_ratio: Fraction of data for out-of-sample testing (e.g. 0.2).
            purge_gap: Number of bars to discard before split boundaries to prevent
                forward target overlap from bleeding into subsequent folds.

        Returns:
            ChronologicalSplit with array indices and timestamps.
        """
        total_ratio = train_ratio + val_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0, atol=1e-5):
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total_ratio:.4f} "
                f"({train_ratio} + {val_ratio} + {test_ratio})"
            )

        n = len(timestamps)
        if n < 3:
            raise ValueError(f"Insufficient data for splitting: {n} observations")

        # Verify timestamps are sorted chronologically
        ts_arr = np.asarray(timestamps)
        if not np.all(ts_arr[:-1] <= ts_arr[1:]):
            raise ValueError(
                "Timestamps must be sorted in ascending chronological order "
                "before splitting."
            )

        train_cutoff = int(n * train_ratio)
        val_cutoff = int(n * (train_ratio + val_ratio))

        # Enforce minimum sizes
        if train_cutoff <= 0:
            raise ValueError(f"Train split is empty with ratio {train_ratio}")
        if val_ratio > 0 and val_cutoff <= train_cutoff:
            raise ValueError(f"Validation split is empty with ratio {val_ratio}")
        if test_cutoff_size := n - val_cutoff <= 0:
            raise ValueError(f"Test split is empty ({test_cutoff_size} observations)")

        # Apply purge gap
        train_end_idx = max(1, train_cutoff - purge_gap)
        train_indices = np.arange(0, train_end_idx)

        if val_ratio > 0:
            val_indices_raw = np.arange(train_cutoff, val_cutoff)
            if purge_gap > 0 and len(val_indices_raw) > purge_gap:
                val_indices = val_indices_raw[:-purge_gap]
            else:
                val_indices = val_indices_raw
        else:
            val_indices = np.array([], dtype=int)

        test_indices = np.arange(val_cutoff, n)

        def _format_ts(val: Any) -> str:
            if hasattr(val, "isoformat"):
                return val.isoformat()
            return str(val)

        return ChronologicalSplit(
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            train_start=_format_ts(ts_arr[train_indices[0]]),
            train_end=_format_ts(ts_arr[train_indices[-1]]),
            val_start=(
                _format_ts(ts_arr[val_indices[0]]) if len(val_indices) > 0 else None
            ),
            val_end=(
                _format_ts(ts_arr[val_indices[-1]]) if len(val_indices) > 0 else None
            ),
            test_start=_format_ts(ts_arr[test_indices[0]]),
            test_end=_format_ts(ts_arr[test_indices[-1]]),
            purge_gap=purge_gap,
        )

    @staticmethod
    def split_by_date(
        timestamps: np.ndarray | list[Any],
        train_end: str,
        val_end: str | None = None,
        train_start: str | None = None,
        purge_gap: int = 0,
    ) -> ChronologicalSplit:
        """Split ordered data using explicit date cutoffs.

        Args:
            timestamps: 1D array of timestamps corresponding to data rows.
            train_end: End cutoff for training data (inclusive).
            val_end: Optional end cutoff for validation data (inclusive).
            train_start: Optional start cutoff for training data.
            purge_gap: Number of bars to purge before fold transitions.

        Returns:
            ChronologicalSplit with array indices and timestamps.
        """
        ts_arr = np.asarray(timestamps)
        n = len(ts_arr)
        if n < 3:
            raise ValueError(f"Insufficient data for splitting: {n} observations")

        # Convert string representations for comparison if needed
        ts_strings = np.array(
            [t.isoformat() if hasattr(t, "isoformat") else str(t) for t in ts_arr]
        )

        train_mask = ts_strings <= train_end
        if train_start is not None:
            train_mask &= ts_strings >= train_start

        train_idx = np.where(train_mask)[0]
        if len(train_idx) == 0:
            raise ValueError(f"No records match training period <= {train_end}")

        if val_end is not None:
            val_mask = (ts_strings > train_end) & (ts_strings <= val_end)
            val_idx = np.where(val_mask)[0]
            test_mask = ts_strings > val_end
            test_idx = np.where(test_mask)[0]
        else:
            val_idx = np.array([], dtype=int)
            test_mask = ts_strings > train_end
            test_idx = np.where(test_mask)[0]

        if len(test_idx) == 0:
            raise ValueError("No records match test period after cutoffs")

        # Apply purge gap
        if purge_gap > 0 and len(train_idx) > purge_gap:
            train_idx = train_idx[:-purge_gap]
        if purge_gap > 0 and len(val_idx) > purge_gap:
            val_idx = val_idx[:-purge_gap]

        def _format_ts(val: Any) -> str:
            if hasattr(val, "isoformat"):
                return val.isoformat()
            return str(val)

        return ChronologicalSplit(
            train_indices=train_idx,
            val_indices=val_idx,
            test_indices=test_idx,
            train_start=_format_ts(ts_arr[train_idx[0]]),
            train_end=_format_ts(ts_arr[train_idx[-1]]),
            val_start=_format_ts(ts_arr[val_idx[0]]) if len(val_idx) > 0 else None,
            val_end=_format_ts(ts_arr[val_idx[-1]]) if len(val_idx) > 0 else None,
            test_start=_format_ts(ts_arr[test_idx[0]]),
            test_end=_format_ts(ts_arr[test_idx[-1]]),
            purge_gap=purge_gap,
        )
