"""Parquet storage and DuckDB analytical query layer for TRADY."""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from trady.data.metadata import DatasetMetadata
from trady.data.normalizer import arrow_table_to_records, records_to_arrow_table
from trady.data.schema import OHLCVRecord
from trady.utils.logging import get_logger

logger = get_logger("data.storage")


def calculate_file_sha256(file_path: Path | str) -> str:
    """Compute the SHA-256 hash of a file for immutability verification."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_dataset(
    dataset: pa.Table | Sequence[OHLCVRecord],
    destination_path: Path | str,
    metadata: DatasetMetadata | None = None,
) -> Path:
    """Save an OHLCV dataset as an optimized Parquet file with metadata.

    Args:
        dataset: A PyArrow Table or sequence of OHLCVRecord objects.
        destination_path: Destination path ending in .parquet.
        metadata: Optional DatasetMetadata container.

    Returns:
        Path to the saved Parquet file.
    """
    dest = Path(destination_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(dataset, pa.Table):
        table = dataset
    else:
        table = records_to_arrow_table(dataset)

    # Attach JSON-encoded metadata to Parquet schema metadata
    if metadata is not None:
        custom_metadata = {
            b"trady_metadata": json.dumps(metadata.to_dict()).encode("utf-8")
        }
        existing_meta = table.schema.metadata or {}
        existing_meta.update(custom_metadata)
        table = table.replace_schema_metadata(existing_meta)

    # Write Parquet with Snappy compression and standard dictionary encoding
    pq.write_table(
        table,
        dest,
        compression="snappy",
        use_dictionary=True,
    )
    logger.info("Saved %d records to %s", table.num_rows, dest)

    # Also persist companion .meta.json file if metadata is provided
    if metadata is not None:
        meta_file = dest.with_suffix(".meta.json")
        # Update metadata checksum with the generated file hash
        file_hash = calculate_file_sha256(dest)
        updated_meta = DatasetMetadata(
            provider=metadata.provider,
            symbol=metadata.symbol,
            timeframe=metadata.timeframe,
            start_time=metadata.start_time,
            end_time=metadata.end_time,
            timezone=metadata.timezone,
            retrieval_time=metadata.retrieval_time,
            schema_version=metadata.schema_version,
            record_count=table.num_rows,
            checksum_sha256=file_hash,
            extra=metadata.extra,
        )
        updated_meta.save(meta_file)

    return dest


def load_dataset(source_path: Path | str) -> pa.Table:
    """Read a Parquet dataset into a PyArrow Table."""
    src = Path(source_path)
    if not src.is_file():
        raise FileNotFoundError(f"Dataset file not found: {src}")
    return pq.read_table(src)


def load_records(source_path: Path | str) -> list[OHLCVRecord]:
    """Read a Parquet dataset and deserialize into canonical OHLCVRecord objects."""
    table = load_dataset(source_path)
    return arrow_table_to_records(table)


def inspect_dataset(file_path: Path | str) -> dict[str, Any]:
    """Inspect dataset schema, row count, byte size, date bounds, and metadata."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    file_size_bytes = path.stat().st_size
    parquet_file = pq.ParquetFile(path)
    schema = parquet_file.schema_arrow
    num_rows = parquet_file.metadata.num_rows

    # Extract schema metadata if present
    custom_meta: dict[str, Any] = {}
    if schema.metadata and b"trady_metadata" in schema.metadata:
        try:
            custom_meta = json.loads(schema.metadata[b"trady_metadata"].decode("utf-8"))
        except Exception:
            pass

    # Read companion .meta.json if present
    meta_json_path = path.with_suffix(".meta.json")
    if not custom_meta and meta_json_path.is_file():
        try:
            with open(meta_json_path, encoding="utf-8") as f:
                custom_meta = json.load(f)
        except Exception:
            pass

    # Analytical inspection via DuckDB for accurate min/max timestamps and symbol list
    rel_path_str = str(path).replace("\\", "/")
    con = duckdb.connect(database=":memory:")
    try:
        query = (
            f"SELECT symbol, MIN(timestamp) as min_ts, MAX(timestamp) as max_ts, "
            f"COUNT(*) as cnt FROM '{rel_path_str}' GROUP BY symbol"
        )
        summary_rows = con.execute(query).fetchall()
        symbols = [r[0] for r in summary_rows]
        min_ts = min([r[1] for r in summary_rows]) if summary_rows else None
        max_ts = max([r[2] for r in summary_rows]) if summary_rows else None
    finally:
        con.close()

    return {
        "file_path": str(path),
        "file_size_bytes": file_size_bytes,
        "record_count": num_rows,
        "columns": schema.names,
        "symbols": symbols,
        "start_time": str(min_ts) if min_ts else "N/A",
        "end_time": str(max_ts) if max_ts else "N/A",
        "metadata": custom_meta,
    }


def query_dataset(
    query_sql: str,
    con: duckdb.DuckDBPyConnection | None = None,
) -> list[dict[str, Any]]:
    """Execute analytical SQL query using DuckDB without pandas dependency.

    Example:
        results = query_dataset("SELECT * FROM 'data/processed/SPY_1d.parquet'")
    """
    should_close = False
    if con is None:
        con = duckdb.connect(database=":memory:")
        should_close = True

    try:
        rel = con.execute(query_sql)
        if rel.description is None:
            return []
        cols = [c[0] for c in rel.description]
        rows = rel.fetchall()
        return [dict(zip(cols, row, strict=True)) for row in rows]
    finally:
        if should_close:
            con.close()
