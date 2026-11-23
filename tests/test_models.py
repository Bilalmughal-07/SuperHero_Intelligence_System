from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from superhero_ml.config import FEATURES
from superhero_ml.data import prepare_datasets
from superhero_ml.models import run_classification, run_clustering


def _write_synthetic_dataset(tmp_path):
    rng = np.random.default_rng(42)
    train_rows = []
    test_rows = []
    for i in range(80):
        row = {"name": f"train-{i}", "alignment": "Good" if i % 2 == 0 else "Bad"}
        row.update({feature: int(rng.integers(1, 101)) for feature in FEATURES})
        train_rows.append(row)
    for i in range(40):
        row = {"name": f"test-{i}", "alignment": "Good" if i % 2 == 0 else "Bad"}
        row.update({feature: int(rng.integers(1, 101)) for feature in FEATURES})
        test_rows.append(row)
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    pd.DataFrame(train_rows).to_csv(train_path, index=False)
    pd.DataFrame(test_rows).to_csv(test_path, index=False)
    return train_path, test_path


def test_classification_and_clustering_smoke(tmp_path):
    train_path, test_path = _write_synthetic_dataset(tmp_path)
    bundle = prepare_datasets(train_path, test_path)
    classification = run_classification(bundle)
    clustering = run_clustering(bundle)

    assert classification["grid_sizes"] == {"Decision Tree": 15, "KNN": 12}
    assert set(classification["evaluations"]) == {"Baseline", "Decision Tree", "KNN"}
    assert clustering["sweep"]["k"].tolist() == list(range(2, 9))
    assert clustering["k"] in range(2, 9)

