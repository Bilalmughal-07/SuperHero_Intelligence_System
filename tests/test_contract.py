from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from superhero_ml.config import FEATURES, PROHIBITED_FEATURES
from superhero_ml.validation import validate_feature_contract
from run_assignment import _rank_final_table

import pandas as pd


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


def test_final_ranking_uses_villain_recall_only_for_close_macro_f1():
    frame = pd.DataFrame([
        {"model": "Lower F1", "test_accuracy": 0.70, "test_macro_f1": 0.500, "villain_recall": 0.10, "beats_baseline": "yes"},
        {"model": "Higher recall", "test_accuracy": 0.69, "test_macro_f1": 0.495, "villain_recall": 0.30, "beats_baseline": "yes"},
        {"model": "Clearly lower", "test_accuracy": 0.68, "test_macro_f1": 0.470, "villain_recall": 0.90, "beats_baseline": "yes"},
    ])
    ranked = _rank_final_table(frame)
    assert ranked.iloc[0]["model"] == "Higher recall"
    assert ranked.iloc[1]["model"] == "Lower F1"
    assert ranked.iloc[2]["model"] == "Clearly lower"

