from __future__ import annotations

from typing import Any

from .config import FEATURES, PROHIBITED_FEATURES, SEED


def validate_feature_contract(feature_names) -> dict[str, Any]:
    names = list(feature_names)
    return {
        "exact_six_features": names == FEATURES,
        "prohibited_features_absent": not (set(names) & PROHIBITED_FEATURES),
        "feature_count": len(names),
    }


def validate_run(classification: dict, clustering: dict, neural: dict, artifact_paths: list[str]) -> dict[str, Any]:
    checks: dict[str, Any] = {
        "seed": SEED == 42,
        "feature_contract": validate_feature_contract(FEATURES),
        "decision_tree_grid_size": classification["grid_sizes"]["Decision Tree"] == 15,
        "knn_grid_size": classification["grid_sizes"]["KNN"] == 12,
        "clustering_k_values": clustering["sweep"]["k"].tolist() == list(range(2, 9)),
        "selected_k_in_range": int(clustering["k"]) in range(2, 9),
        "network_a_parameters": neural["networks"]["Network A"]["parameters"] == 65,
        "network_b_parameters": neural["networks"]["Network B"]["parameters"] == 257,
        "all_graphs_and_tables_generated": len(artifact_paths) >= 27,
    }
    checks["all_passed"] = all(
        value if isinstance(value, bool) else all(value.values()) if isinstance(value, dict) else True
        for value in checks.values()
    )
    if not checks["all_passed"]:
        failed = [key for key, value in checks.items() if key != "all_passed" and value is False]
        raise AssertionError(f"Acceptance checks failed: {failed}")
    return checks

