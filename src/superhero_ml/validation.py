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


def validate_run(
    classification: dict,
    clustering: dict,
    neural: dict,
    artifact_paths: list[str],
    artifact_captions: dict[str, str] | None = None,
    generated_tables: dict[str, Any] | None = None,
) -> dict[str, Any]:
    generated_tables = generated_tables or {}
    required_caption_keys = {f"G{i}" for i in range(1, 17)} | {f"T{i}" for i in range(1, 12)}
    caption_check = artifact_captions is not None and required_caption_keys.issubset(artifact_captions)
    profile_check = all("Name" in frame.columns for frame in (clustering["kmeans_profiles"], clustering["agglomerative_profiles"]))
    required_nn_metrics = {"hero_precision", "hero_recall", "hero_f1", "villain_precision", "villain_recall", "villain_f1"}
    nn_metric_check = all(required_nn_metrics.issubset(neural["networks"][name]["metrics"]["test"]) for name in ("Network A", "Network B"))
    table_check = {key: key in generated_tables for key in [f"T{i}" for i in range(1, 12)]}
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
        "cluster_profile_names": profile_check,
        "nn_per_class_metrics": nn_metric_check,
        "artifact_captions": caption_check,
        "all_required_tables_present": table_check,
    }
    checks["all_passed"] = all(
        value if isinstance(value, bool) else all(value.values()) if isinstance(value, dict) else True
        for value in checks.values()
    )
    if not checks["all_passed"]:
        failed = [key for key, value in checks.items() if key != "all_passed" and value is False]
        raise AssertionError(f"Acceptance checks failed: {failed}")
    return checks

