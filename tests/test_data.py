from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from superhero_ml.config import FEATURES
from superhero_ml.data import prepare_datasets


def _frame(names, alignments):
    rows = []
    for i, (name, alignment) in enumerate(zip(names, alignments)):
        row = {"name": name, "alignment": alignment}
        row.update({feature: i + 1 for feature in FEATURES})
        rows.append(row)
    return pd.DataFrame(rows)


def test_cleaning_converts_zero_and_blank_and_removes_weak_rows(tmp_path):
    train = _frame(["A", "B", "C", "D"], ["Good", "Bad", "Neutral", "Good"])
    test = _frame(["E", "F"], ["Good", "Bad"])
    train["speed_score"] = train["speed_score"].astype(object)
    train.loc[0, "strength_score"] = 0
    train.loc[0, "speed_score"] = ""
    train.loc[2, FEATURES[:3]] = 0
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    bundle = prepare_datasets(train_path, test_path)

    assert len(bundle.train_clean_a) == 3
    assert bundle.train_clean_a.loc[bundle.train_clean_a["name"] == "A", "strength_score"].isna().all()
    assert "C" not in set(bundle.train_clean_a["name"])
    assert set(bundle.train_clean_b["target"]) == {0, 1}


def test_cross_file_overlap_is_removed_from_training(tmp_path):
    train = _frame(["Shared", "TrainOnly"], ["Good", "Bad"])
    test = _frame(["Shared", "TestOnly"], ["Bad", "Good"])
    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)

    bundle = prepare_datasets(tmp_path / "train.csv", tmp_path / "test.csv")

    assert "Shared" not in set(bundle.train_clean_a["name"])
    assert "Shared" in set(bundle.test_clean_b["name"])

