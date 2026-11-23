from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from superhero_ml.config import FEATURES, PROHIBITED_FEATURES
from superhero_ml.validation import validate_feature_contract


def test_exact_feature_contract():
    result = validate_feature_contract(FEATURES)
    assert result["exact_six_features"] is True
    assert result["prohibited_features_absent"] is True
    assert result["feature_count"] == 6


def test_prohibited_feature_is_rejected():
    result = validate_feature_contract(FEATURES + ["overall_score"])
    assert result["exact_six_features"] is False
    assert result["prohibited_features_absent"] is False
    assert "overall_score" in PROHIBITED_FEATURES

