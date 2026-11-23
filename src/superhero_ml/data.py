from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import FEATURES, LABEL_MAP, NAME_COLUMN, TARGET


@dataclass
class DatasetBundle:
    train_raw: pd.DataFrame
    test_raw: pd.DataFrame
    train_clean_a: pd.DataFrame
    train_clean_b: pd.DataFrame
    test_clean_b: pd.DataFrame
    audit: pd.DataFrame
    row_counts: pd.DataFrame


def _require_columns(frame: pd.DataFrame, file_name: str) -> None:
    required = set(FEATURES) | {NAME_COLUMN, TARGET}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{file_name} is missing required columns: {missing}")


def audit_frame(frame: pd.DataFrame, file_name: str) -> dict[str, Any]:
    _require_columns(frame, file_name)
    record: dict[str, Any] = {
        "file": file_name,
        "rows": len(frame),
        "columns": len(frame.columns),
        "column_names": json.dumps(list(frame.columns)),
        "dtypes": json.dumps({column: str(dtype) for column, dtype in frame.dtypes.items()}),
        "missing_by_column": json.dumps({column: int(count) for column, count in frame.isna().sum().items()}),
        "duplicate_names": int(frame[NAME_COLUMN].duplicated(keep=False).sum()),
        "alignment_good": int(frame[TARGET].eq("Good").sum()),
        "alignment_bad": int(frame[TARGET].eq("Bad").sum()),
        "alignment_neutral": int(frame[TARGET].eq("Neutral").sum()),
        "alignment_blank": int(frame[TARGET].isna().sum() + frame[TARGET].astype("string").str.strip().eq("").sum()),
        "missing_total": int(frame.isna().sum().sum()),
    }
    for column in FEATURES:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        record[f"missing_{column}"] = int(frame[column].isna().sum())
        record[f"zeros_{column}"] = int(numeric.eq(0).sum())
        record[f"blanks_{column}"] = int(frame[column].astype("string").str.strip().eq("").sum())
    return record


def _deduplicate(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    before = len(frame)
    duplicate_mask = frame[NAME_COLUMN].duplicated(keep="first")
    duplicate_removed = int(duplicate_mask.sum())
    return frame.loc[~duplicate_mask].copy(), {
        "loaded": before,
        "duplicates_removed": duplicate_removed,
        "after_duplicates": before - duplicate_removed,
    }


def _mark_missing_and_remove_weak(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    conversion_counts: dict[str, int] = {}

    for column in FEATURES:
        original = frame[column]
        blank_mask = original.astype("string").str.strip().eq("")
        numeric = pd.to_numeric(original.mask(blank_mask, np.nan), errors="coerce")
        conversion_counts[f"zeros_to_nan_{column}"] = int(numeric.eq(0).sum())
        conversion_counts[f"blanks_to_nan_{column}"] = int(blank_mask.sum())
        frame[column] = numeric.mask(numeric.eq(0), np.nan)

    missing_per_row = frame[FEATURES].isna().sum(axis=1)
    weak_mask = missing_per_row.ge(3)
    weak_removed = int(weak_mask.sum())
    frame = frame.loc[~weak_mask].copy()
    counts = {
        "weak_rows_removed": weak_removed,
        "after_weak_rows": len(frame),
        **conversion_counts,
    }
    return frame, counts


def prepare_datasets(train_path: str | Path, test_path: str | Path) -> DatasetBundle:
    """Load each raw file once and produce the two assignment clean sets.

    The test data is never imputed here. Model-specific imputers are fitted on
    permitted training data and only transform the frozen cleaned test frame.
    """
    train_path = Path(train_path)
    test_path = Path(test_path)
    if not train_path.exists() or not test_path.exists():
        missing = [str(path) for path in (train_path, test_path) if not path.exists()]
        raise FileNotFoundError("Missing required assignment data: " + ", ".join(missing))

    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    train_audit = audit_frame(train_raw, train_path.name)
    test_audit = audit_frame(test_raw, test_path.name)

    train_dedup, train_counts = _deduplicate(train_raw)
    test_dedup, test_counts = _deduplicate(test_raw)

    overlapping_names = set(train_dedup[NAME_COLUMN].dropna()) & set(test_dedup[NAME_COLUMN].dropna())
    if overlapping_names:
        train_dedup = train_dedup.loc[~train_dedup[NAME_COLUMN].isin(overlapping_names)].copy()
    train_counts["cross_file_overlap_removed"] = len(overlapping_names)
    test_counts["cross_file_overlap_removed"] = 0
    train_counts["after_cross_file_overlap"] = len(train_dedup)
    test_counts["after_cross_file_overlap"] = len(test_dedup)

    train_clean, train_missing_counts = _mark_missing_and_remove_weak(train_dedup)
    test_clean, test_missing_counts = _mark_missing_and_remove_weak(test_dedup)
    train_counts.update(train_missing_counts)
    test_counts.update(test_missing_counts)

    train_clean_a = train_clean.copy()
    train_clean_b = train_clean.loc[train_clean[TARGET].isin(LABEL_MAP)].copy()
    test_clean_b = test_clean.loc[test_clean[TARGET].isin(LABEL_MAP)].copy()
    train_clean_b["target"] = train_clean_b[TARGET].map(LABEL_MAP).astype(int)
    test_clean_b["target"] = test_clean_b[TARGET].map(LABEL_MAP).astype(int)

    train_counts["clean_set_a"] = len(train_clean_a)
    train_counts["clean_set_b"] = len(train_clean_b)
    test_counts["clean_set_a"] = np.nan
    test_counts["clean_set_b"] = len(test_clean_b)

    audit = pd.DataFrame([train_audit, test_audit])
    row_counts = pd.DataFrame([
        {"stage": "Loaded", "train.csv": train_counts["loaded"], "test.csv": test_counts["loaded"]},
        {"stage": "After step 2 duplicates", "train.csv": train_counts["after_duplicates"], "test.csv": test_counts["after_duplicates"]},
        {"stage": "After cross-file duplicate rule", "train.csv": train_counts["after_cross_file_overlap"], "test.csv": test_counts["after_cross_file_overlap"]},
        {"stage": "After step 4 weak rows", "train.csv": train_counts["after_weak_rows"], "test.csv": test_counts["after_weak_rows"]},
        {"stage": "Clean Set A", "train.csv": train_counts["clean_set_a"], "test.csv": np.nan},
        {"stage": "Clean Set B", "train.csv": train_counts["clean_set_b"], "test.csv": test_counts["clean_set_b"]},
    ])
    return DatasetBundle(train_raw, test_raw, train_clean_a, train_clean_b, test_clean_b, audit, row_counts)

