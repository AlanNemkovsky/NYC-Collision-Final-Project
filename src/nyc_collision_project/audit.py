from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import ProjectConfig
from .io_utils import read_header, read_csv_chunks, write_json, write_table, write_text
from .schema import normalize_col, get_alias


def _table_audit(path: Path, table_name: str, chunk_size: int) -> tuple[dict, pd.DataFrame]:
    header = read_header(path)
    row_count = 0
    missing_counts = pd.Series(0, index=header, dtype="int64")
    collision_col = get_alias(header, "collision_id", required=False)
    unique_collision_estimate = set()
    unique_cap = 2_000_000

    for chunk in read_csv_chunks(path, chunksize=chunk_size):
        row_count += len(chunk)
        missing_counts = missing_counts.add(chunk.isna().sum(), fill_value=0).astype("int64")
        if collision_col and collision_col in chunk.columns and len(unique_collision_estimate) < unique_cap:
            vals = chunk[collision_col].dropna().astype(str).unique()
            unique_collision_estimate.update(vals[: max(0, unique_cap - len(unique_collision_estimate))])

    miss = pd.DataFrame({
        "table": table_name,
        "column": header,
        "normalized_column": [normalize_col(c) for c in header],
        "missing_count": [int(missing_counts.get(c, 0)) for c in header],
        "missing_rate": [float(missing_counts.get(c, 0) / row_count) if row_count else 0.0 for c in header],
    })
    summary = {
        "table": table_name,
        "path": str(path),
        "row_count": row_count,
        "column_count": len(header),
        "columns": header,
        "collision_id_column_detected": collision_col,
        "collision_id_unique_count_sample_or_full": len(unique_collision_estimate),
        "collision_id_unique_cap_reached": len(unique_collision_estimate) >= unique_cap,
    }
    return summary, miss


def run_raw_audit(config: ProjectConfig) -> dict:
    reports_dir = config.outputs_dir / "reports"
    tables_dir = config.outputs_dir / "tables"
    latex_dir = config.outputs_dir / "latex_tables"

    summaries = []
    missing_frames = []
    for name, path in [
        ("crashes", config.crashes_path),
        ("vehicles", config.vehicles_path),
        ("persons", config.persons_path),
    ]:
        summary, miss = _table_audit(path, name, config.chunk_size)
        summaries.append(summary)
        missing_frames.append(miss)

    missing_all = pd.concat(missing_frames, ignore_index=True)
    write_table(missing_all, tables_dir / "01_missingness_by_table.csv", latex_dir / "01_missingness_by_table.tex")
    write_json({"tables": summaries}, reports_dir / "01_schema_report.json")

    lines = ["Raw data audit", "==============", ""]
    for s in summaries:
        lines.extend([
            f"Table: {s['table']}",
            f"Path: {s['path']}",
            f"Rows: {s['row_count']:,}",
            f"Columns: {s['column_count']:,}",
            f"Collision ID column detected: {s['collision_id_column_detected']}",
            f"Collision ID unique count sample/full: {s['collision_id_unique_count_sample_or_full']:,}",
            "",
        ])
    write_text("\n".join(lines), reports_dir / "01_schema_report.txt")

    return {"raw_audit": summaries}
