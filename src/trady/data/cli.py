"""CLI subcommand handlers for TRADY Data Engine."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from trady.data.metadata import DatasetMetadata
from trady.data.normalizer import NormalizationConfig, normalize_records
from trady.data.provider import SyntheticDataProvider
from trady.data.storage import inspect_dataset, query_dataset, save_dataset
from trady.data.validator import validate_arrow_table, validate_records
from trady.utils.logging import get_logger

logger = get_logger("cli.data")


def add_data_subparsers(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Attach the 'data' subcommand tree to the TRADY CLI parser."""
    data_parser = subparsers.add_parser(
        "data",
        help="Market data engine (validate, inspect, normalize, query, metadata)",
    )
    data_subparsers = data_parser.add_subparsers(
        dest="data_command", help="Data subcommands"
    )

    # 1. validate
    val_p = data_subparsers.add_parser(
        "validate", help="Validate a dataset against TRADY canonical invariants"
    )
    val_p.add_argument("dataset", type=str, help="Path to Parquet or JSON/CSV dataset")
    val_p.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Optional expected symbol constraint",
    )

    # 2. inspect
    insp_p = data_subparsers.add_parser(
        "inspect", help="Inspect schema, bounds, row count, and size of a dataset"
    )
    insp_p.add_argument("dataset", type=str, help="Path to Parquet dataset file")

    # 3. normalize
    norm_p = data_subparsers.add_parser(
        "normalize", help="Normalize raw data into TRADY canonical Parquet format"
    )
    norm_p.add_argument("input", type=str, help="Path to raw source file (CSV, JSON)")
    norm_p.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output destination path (must end in .parquet)",
    )
    norm_p.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Default ticker symbol if omitted in raw data",
    )
    norm_p.add_argument(
        "--deduplicate",
        action="store_true",
        help="Explicitly discard duplicate timestamp records",
    )

    # 4. metadata
    meta_p = data_subparsers.add_parser(
        "metadata", help="Display metadata for a dataset"
    )
    meta_p.add_argument(
        "dataset", type=str, help="Path to Parquet dataset or .meta.json file"
    )

    # 5. query
    qry_p = data_subparsers.add_parser(
        "query", help="Execute an analytical DuckDB SQL query"
    )
    qry_p.add_argument(
        "sql", type=str, help="SQL query (e.g. SELECT * FROM 'file.parquet' LIMIT 5)"
    )

    # 6. generate-synthetic
    gen_p = data_subparsers.add_parser(
        "generate-synthetic",
        help="Generate deterministic synthetic market data fixture",
    )
    gen_p.add_argument("--symbol", type=str, default="SPY", help="Ticker symbol")
    gen_p.add_argument(
        "--bars", type=int, default=30, help="Number of OHLCV bars to generate"
    )
    gen_p.add_argument(
        "--timeframe", type=str, default="1d", help="Bar timeframe (1d, 1h)"
    )
    gen_p.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/raw/synthetic_SPY_1d.json",
        help="Output file path",
    )


def _load_raw_file(file_path: Path) -> list[dict[str, Any]]:
    """Helper to read JSON or CSV into a list of row dictionaries."""
    suffix = file_path.suffix.lower()
    if suffix == ".json":
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "records" in data:
                return data["records"]
            raise ValueError(
                "JSON file must be an array of records or contain 'records' key"
            )
    elif suffix in (".csv", ".txt"):
        with open(file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)
    elif suffix == ".parquet":
        table = pq.read_table(file_path)
        return table.to_pylist()
    raise ValueError(f"Unsupported file format: {suffix}")


def handle_data_command(args: argparse.Namespace) -> int:
    """Dispatcher for 'trady data <subcommand>' CLI calls."""
    cmd = getattr(args, "data_command", None)
    if not cmd:
        print(
            "Usage: trady data "
            "[validate|inspect|normalize|metadata|query|generate-synthetic] ..."
        )
        return 0

    if cmd == "validate":
        target = Path(args.dataset)
        if not target.is_file():
            print(f"[ERROR] Target file does not exist: {target}")
            return 1

        print(f"Validating dataset: {target}")
        if target.suffix.lower() == ".parquet":
            table = pq.read_table(target)
            res = validate_arrow_table(table, expected_symbol=args.symbol)
        else:
            records = _load_raw_file(target)
            res = validate_records(records, expected_symbol=args.symbol)

        print(res.format_report())
        return 0 if res.is_valid else 1

    elif cmd == "inspect":
        target = Path(args.dataset)
        if not target.is_file():
            print(f"[ERROR] Target file does not exist: {target}")
            return 1
        info = inspect_dataset(target)
        print("=" * 60)
        print(f" DATASET INSPECTION: {target.name}")
        print("=" * 60)
        print(f"  - File Size:     {info['file_size_bytes']} bytes")
        print(f"  - Record Count:  {info['record_count']}")
        print(f"  - Symbols:       {', '.join(info['symbols'])}")
        print(f"  - Date Range:    {info['start_time']} -> {info['end_time']}")
        print(f"  - Columns:       {', '.join(info['columns'])}")
        if info.get("metadata"):
            print("\n  Provenance Metadata:")
            print(json.dumps(info["metadata"], indent=4))
        return 0

    elif cmd == "normalize":
        in_path = Path(args.input)
        out_path = Path(args.output)
        if not in_path.is_file():
            print(f"[ERROR] Input file does not exist: {in_path}")
            return 1

        print(f"Normalizing '{in_path}' -> '{out_path}'...")
        raw_rows = _load_raw_file(in_path)
        cfg = NormalizationConfig(
            deduplicate=args.deduplicate,
            default_symbol=args.symbol,
        )
        records, logs = normalize_records(raw_rows, config=cfg)
        for lg in logs:
            print(f"  [NORM] {lg}")

        # Validate normalized output before writing
        val_res = validate_records(records)
        if not val_res.is_valid:
            print("\n[FAIL] Normalized data failed validation!")
            print(val_res.format_report())
            return 1

        meta = DatasetMetadata(
            provider="normalization_pipeline",
            symbol=records[0].symbol if records else "UNKNOWN",
            timeframe="unknown",
            start_time=records[0].timestamp,
            end_time=records[-1].timestamp,
            record_count=len(records),
            extra={"source_file": str(in_path)},
        )
        saved_file = save_dataset(records, out_path, metadata=meta)
        print(f"\n[OK] Successfully normalized and saved to {saved_file}")
        return 0

    elif cmd == "metadata":
        target = Path(args.dataset)
        if target.suffix.lower() == ".json":
            meta = DatasetMetadata.load(target)
        else:
            info = inspect_dataset(target)
            meta_dict = info.get("metadata")
            if not meta_dict:
                print(f"[WARN] No metadata found for {target}")
                return 1
            meta = DatasetMetadata.from_dict(meta_dict)

        print(json.dumps(meta.to_dict(), indent=2))
        return 0

    elif cmd == "query":
        print(f"Executing Query: {args.sql}\n")
        try:
            results = query_dataset(args.sql)
            if not results:
                print("Query returned 0 rows.")
                return 0
            # Print tabular format
            keys = list(results[0].keys())
            header = " | ".join(f"{k:>15}" for k in keys)
            print(header)
            print("-" * len(header))
            for row in results[:20]:
                row_str = " | ".join(f"{str(row[k]):>15}" for k in keys)
                print(row_str)
            if len(results) > 20:
                print(f"\n... ({len(results) - 20} more rows)")
            return 0
        except Exception as exc:
            print(f"[ERROR] Query failed: {exc}")
            return 1

    elif cmd == "generate-synthetic":
        provider = SyntheticDataProvider()
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        records, meta = provider.generate_valid_dataset(
            symbol=args.symbol,
            timeframe=args.timeframe,
            num_bars=args.bars,
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"records": records, "metadata": meta.to_dict()}, f, indent=2)
        print(f"[OK] Generated {len(records)} synthetic bars saved to {out_path}")
        return 0

    return 0
