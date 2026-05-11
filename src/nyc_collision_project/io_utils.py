from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_parent(path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_header(path: Path) -> list[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find input CSV: {path}")
    return list(pd.read_csv(path, nrows=0).columns)


def safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Read CSV with a fallback for encoding edge cases."""
    try:
        return pd.read_csv(path, low_memory=False, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="latin1", **kwargs)


def read_csv_chunks(path: Path, chunksize: int, usecols=None):
    try:
        yield from pd.read_csv(path, chunksize=chunksize, usecols=usecols, low_memory=False)
    except UnicodeDecodeError:
        yield from pd.read_csv(path, chunksize=chunksize, usecols=usecols, low_memory=False, encoding="latin1")


def write_json(obj: dict[str, Any], path: Path) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def write_text(text: str, path: Path) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def write_table(df: pd.DataFrame, path: Path, latex_path: Path | None = None, index: bool = False) -> None:
    ensure_parent(path)
    df.to_csv(path, index=index)
    if latex_path is not None:
        ensure_parent(latex_path)
        with open(latex_path, "w", encoding="utf-8") as f:
            f.write(df.to_latex(index=index, escape=False, longtable=False))


def write_parquet_or_pickle(df: pd.DataFrame, path: Path) -> Path:
    ensure_parent(path)
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception:
        fallback = path.with_suffix(".pkl")
        df.to_pickle(fallback)
        return fallback


def read_parquet_or_pickle(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    fallback = path.with_suffix(".pkl")
    if fallback.exists():
        return pd.read_pickle(fallback)
    raise FileNotFoundError(f"Could not find {path} or {fallback}")


def make_zip(src_dir: Path, zip_base: Path) -> Path:
    zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=src_dir)
    return Path(zip_path)
