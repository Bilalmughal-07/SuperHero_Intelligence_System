from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from superhero_ml.metrics import classification_metrics


def test_classification_metrics_use_required_class_order():
    result = classification_metrics([0, 0, 1, 1], [0, 1, 0, 1], [0.1, 0.8, 0.2, 0.9])
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["macro_f1"] == 0.5
    assert "villain_recall" in result

