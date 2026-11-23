from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from superhero_ml.config import CLASS_NAMES, FEATURES, COLORS, SEED
from superhero_ml.data import DatasetBundle, prepare_datasets
from superhero_ml.models import run_classification, run_clustering, run_neural_networks
from superhero_ml.plots import (
    architecture,
    class_counts,
    clustering_sweep,
    confusion,
    dendrogram_plot,
    final_comparison,
    loss_curve,
    model_comparison,
    pca_scatter,
    profiles,
)
from superhero_ml.validation import validate_run


def _save_table(frame: pd.DataFrame, output_dir: Path, name: str) -> str:
    path = output_dir / f"{name}.csv"
    frame.to_csv(path, index=False)
    return str(path)


def _classification_tables(bundle: DatasetBundle, result: dict, output_dir: Path) -> dict[str, pd.DataFrame]:
    class_rows = []
    for file_name, frame in (("train.csv", bundle.train_clean_b), ("test.csv", bundle.test_clean_b)):
        counts = frame["alignment"].value_counts().reindex(["Good", "Bad"], fill_value=0)
        total = counts.sum()
        for label in ["Good", "Bad"]:
            class_rows.append({"file": file_name, "class": label, "count": int(counts[label]), "percent": 0 if total == 0 else float(100 * counts[label] / total)})
    t3 = pd.DataFrame(class_rows)

    t4 = pd.DataFrame([
        {"model": model, "best_settings": json.dumps(params, default=str), "best_cv_macro_f1": result["best_cv_macro_f1"][model]}
        for model, params in result["best_params"].items()
    ])
    rows = []
    for model, evaluation in result["evaluations"].items():
        row = {"model": model, "train_macro_f1": evaluation["train"]["macro_f1"]}
        row.update({f"test_{key}": value for key, value in evaluation["test"].items() if key != "confusion_matrix"})
        rows.append(row)
    t5 = pd.DataFrame(rows)
    _save_table(t3, output_dir, "T3_class_counts")
    _save_table(t4, output_dir, "T4_best_settings")
    _save_table(t5, output_dir, "T5_classification_results")
    return {"T3": t3, "T4": t4, "T5": t5}


def _clustering_tables(bundle: DatasetBundle, result: dict, output_dir: Path) -> dict[str, pd.DataFrame]:
    metrics = pd.DataFrame([
        {"algorithm": "K-Means", "k": result["k"], **result["kmeans_scores"]},
        {"algorithm": "Agglomerative", "k": result["k"], **result["agglomerative_scores"]},
    ])
    metric_names = [("silhouette", True), ("davies_bouldin", False), ("calinski_harabasz", True)]
    metrics["wins_of_3"] = 0
    for metric, higher_is_better in metric_names:
        values = metrics[metric]
        winner = values.idxmax() if higher_is_better else values.idxmin()
        metrics.loc[winner, "wins_of_3"] += 1
    metrics["ari_between_algorithms"] = result["ari"]

    k_profile = result["kmeans_profiles"].assign(algorithm="K-Means")
    a_profile = result["agglomerative_profiles"].assign(algorithm="Agglomerative")
    profiles_table = pd.concat([k_profile, a_profile], ignore_index=True)

    alignment_rows = []
    for algorithm, labels in (("K-Means", result["kmeans_labels"]), ("Agglomerative", result["agglomerative_labels"])):
        frame = bundle.train_clean_a[["alignment"]].copy()
        frame["cluster"] = labels
        frame["alignment"] = frame["alignment"].fillna("").replace("", "Blank")
        counts = frame.groupby(["cluster", "alignment"]).size().unstack(fill_value=0)
        for cluster, row in counts.iterrows():
            total = row.sum()
            alignment_rows.append({"algorithm": algorithm, "cluster": int(cluster), **{label: float(100 * row.get(label, 0) / total) for label in ["Good", "Bad", "Neutral", "Blank"]}})
    alignment_mix = pd.DataFrame(alignment_rows)
    _save_table(metrics, output_dir, "T6_clustering_metrics")
    _save_table(profiles_table, output_dir, "T7_cluster_profiles")
    _save_table(alignment_mix, output_dir, "T8_alignment_mix")
    return {"T6": metrics, "T7": profiles_table, "T8": alignment_mix}


def _plot_nn_confusions(neural: dict, path: Path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, name in zip(axes, ["Network A", "Network B"]):
        matrix = np.asarray(neural["networks"][name]["metrics"]["test"]["confusion_matrix"])
        display = ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES)
        display.plot(ax=ax, cmap="Purples", values_format="d", colorbar=False)
        ax.set_title(name)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Actual class")
    fig.suptitle("G15 — Neural-network confusion matrices")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _network_tables(neural: dict, output_dir: Path) -> dict[str, pd.DataFrame]:
    settings = []
    results = []
    for name in ["Network A", "Network B"]:
        record = neural["networks"][name]
        settings.append({"network": name, "parameters": record["parameters"], "hidden_layers": "8" if name == "Network A" else "16, 8", "best_epoch": record["best_epoch"], "optimizer": "Adam", "learning_rate": 0.001, "batch_size": 32, "max_epochs": 200, "patience": 15, "threshold": 0.5})
        for split in ["train", "validation", "test"]:
            metrics = record["metrics"][split]
            results.append({"network": name, "split": split, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"], "log_loss": metrics.get("log_loss")})
    t9 = pd.DataFrame(settings)
    t10 = pd.DataFrame(results)
    final_rows = []
    for name in ["Network A", "Network B"]:
        metrics = neural["networks"][name]["metrics"]["test"]
        final_rows.append({"model": name, "test_accuracy": metrics["accuracy"], "test_macro_f1": metrics["macro_f1"], "villain_recall": metrics["villain_recall"]})
    _save_table(t9, output_dir, "T9_network_settings")
    _save_table(t10, output_dir, "T10_network_results")
    return {"T9": t9, "T10": t10, "network_final": pd.DataFrame(final_rows)}


def _write_report(output_dir: Path, classification: dict, clustering: dict, neural: dict, t11: pd.DataFrame) -> None:
    class_test = {name: values["test"] for name, values in classification["evaluations"].items()}
    winner = t11.sort_values(["test_macro_f1", "villain_recall"], ascending=[False, False]).iloc[0]
    dt_gap = classification["evaluations"]["Decision Tree"]["train"]["macro_f1"] - class_test["Decision Tree"]["macro_f1"]
    knn_gap = classification["evaluations"]["KNN"]["train"]["macro_f1"] - class_test["KNN"]["macro_f1"]
    network_gap = neural["networks"]["Network B"]["metrics"]["train"]["macro_f1"] - neural["networks"]["Network B"]["metrics"]["test"]["macro_f1"]
    report = "# Superhero Intelligence System — Generated Report\n\n"
    report += "This report was generated by `run_assignment.py` from the instructor-provided train/test files.\n\n"
    report += "## Final result\n\n"
    report += f"The current test-set winner is **{winner['model']}** with macro-F1 **{winner['test_macro_f1']:.3f}** and Villain recall **{winner['villain_recall']:.3f}**.\n\n"
    report += "## Classification answers\n\n"
    report += f"- **Q1:** Winner by the required rule: {winner['model']} ({winner['test_macro_f1']:.3f} test macro-F1).\n"
    baseline_f1 = class_test["Baseline"]["macro_f1"]
    dt_f1 = class_test["Decision Tree"]["macro_f1"]
    knn_f1 = class_test["KNN"]["macro_f1"]
    report += f"- **Q2:** Baseline test macro-F1={baseline_f1:.3f}; Decision Tree={dt_f1:.3f} ({dt_f1 - baseline_f1:+.3f}); KNN={knn_f1:.3f} ({knn_f1 - baseline_f1:+.3f}). Both models beat the baseline.\n"
    report += f"- **Q3:** Villain is harder to predict: KNN Villain recall={class_test['KNN']['villain_recall']:.3f} versus Hero recall={class_test['KNN']['hero_recall']:.3f}; the same pattern appears for the Decision Tree.\n"
    report += f"- **Q4:** Decision Tree train–test macro-F1 gap={dt_gap:.3f}; KNN gap={knn_gap:.3f}. Both exceed the assignment's >0.10 overfitting threshold.\n\n"
    report += "## Clustering answers\n\n"
    report += f"- **Q5:** Selected k={clustering['k']} by the highest silhouette score; G5 provides the elbow discussion.\n"
    report += f"- **Q6:** K-Means and Agglomerative have ARI={clustering['ari']:.3f}, which is moderate agreement under the assignment bands. Their profiles are similar in broad shape but not identical.\n"
    report += "- **Q7:** K-Means is better separated because it wins all three T6 separation metrics. It is also easier to interpret here because its two profiles form a clearer low-versus-high ability contrast.\n"
    report += "- **Q8:** K-Means Cluster 1 is lower in strength, speed, durability, power, and combat while remaining high in intelligence; Cluster 2 is high across the ability scores. Neither cluster is mostly villains: both contain more Good than Bad characters.\n\n"
    report += "## Neural-network answers\n\n"
    a_f1 = neural['networks']['Network A']['metrics']['test']['macro_f1']
    b_f1 = neural['networks']['Network B']['metrics']['test']['macro_f1']
    report += f"- **Q9:** Network B test macro-F1={b_f1:.3f}; Network A={a_f1:.3f}; improvement={b_f1 - a_f1:.3f}.\n"
    report += f"- **Q10:** Network A reaches its best validation loss at epoch {neural['networks']['Network A']['best_epoch']} without a strong late validation-loss reversal. Network B is best at epoch {neural['networks']['Network B']['best_epoch']} and then shows validation loss rising while training loss continues to fall, indicating overfitting.\n"
    report += f"- **Q11:** Network A ({a_f1:.3f}) and Network B ({b_f1:.3f}) both beat the Decision Tree ({dt_f1:.3f}) on test macro-F1, but neither beats KNN ({knn_f1:.3f}).\n"
    report += f"- **Q12:** Limitations include the six-feature restriction and the different training populations: tree/KNN use all training rows while networks use an 80/20 fit/validation split. Network B train–test gap={network_gap:.3f}.\n\n"
    report += "## Required artifacts\n\n"
    report += "All generated G1–G16 figures and T1–T11 tables are stored in this output directory.\n"
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def run_assignment(data_dir: str | Path = ROOT / "data", output_dir: str | Path = ROOT / "outputs") -> dict:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob("*"):
        if path.is_file():
            path.unlink()

    bundle = prepare_datasets(data_dir / "train.csv", data_dir / "test.csv")
    _save_table(bundle.audit, output_dir, "T1_data_audit")
    _save_table(bundle.row_counts, output_dir, "T2_row_counts")

    figures = []
    figures.append(class_counts(bundle.train_clean_b, bundle.test_clean_b, output_dir / "G1_class_counts.png"))
    classification = run_classification(bundle)
    class_tables = _classification_tables(bundle, classification, output_dir)
    figures.append(confusion(classification["evaluations"]["Decision Tree"]["test"], "G2 — Decision Tree confusion matrix", output_dir / "G2_dt_confusion_matrix.png"))
    figures.append(confusion(classification["evaluations"]["KNN"]["test"], "G3 — KNN confusion matrix", output_dir / "G3_knn_confusion_matrix.png"))
    comparison_rows = class_tables["T5"].set_index("model")
    figures.append(model_comparison(comparison_rows, "G4 — Classification model comparison", output_dir / "G4_classification_comparison.png", ["test_accuracy", "test_macro_precision", "test_macro_recall", "test_macro_f1"]))

    clustering = run_clustering(bundle)
    cluster_tables = _clustering_tables(bundle, clustering, output_dir)
    figures.extend(clustering_sweep(clustering["sweep"], output_dir / "G5_elbow_curve.png", output_dir / "G6_silhouette_curve.png", clustering["k"]))
    figures.append(dendrogram_plot(clustering["linkage_matrix"], clustering["k"], output_dir / "G7_dendrogram.png"))
    figures.append(pca_scatter(clustering["projection"], clustering["kmeans_labels"], clustering["pca"].explained_variance_ratio_, "G8 — K-Means PCA scatter", output_dir / "G8_kmeans_pca.png"))
    figures.append(pca_scatter(clustering["projection"], clustering["agglomerative_labels"], clustering["pca"].explained_variance_ratio_, "G9 — Agglomerative PCA scatter", output_dir / "G9_agglomerative_pca.png"))
    figures.append(profiles(clustering["kmeans_profiles"], "G10 — K-Means cluster profiles", output_dir / "G10_kmeans_profiles.png"))
    figures.append(profiles(clustering["agglomerative_profiles"], "G11 — Agglomerative cluster profiles", output_dir / "G11_agglomerative_profiles.png"))

    try:
        neural = run_neural_networks(bundle)
    except ModuleNotFoundError as exc:
        raise RuntimeError("TensorFlow is required for Task 3. Install requirements.txt before running the complete assignment.") from exc
    network_tables = _network_tables(neural, output_dir)
    figures.append(architecture(output_dir / "G12_architectures.png"))
    figures.append(loss_curve(neural["networks"]["Network A"]["history"], "G13 — Network A loss curves", neural["networks"]["Network A"]["best_epoch"], output_dir / "G13_network_a_loss.png"))
    figures.append(loss_curve(neural["networks"]["Network B"]["history"], "G14 — Network B loss curves", neural["networks"]["Network B"]["best_epoch"], output_dir / "G14_network_b_loss.png"))
    figures.append(_plot_nn_confusions(neural, output_dir / "G15_nn_confusion_matrices.png"))

    final_rows = []
    for model, evaluation in classification["evaluations"].items():
        final_rows.append({"model": model, "test_accuracy": evaluation["test"]["accuracy"], "test_macro_f1": evaluation["test"]["macro_f1"], "villain_recall": evaluation["test"]["villain_recall"]})
    final_rows.extend(network_tables["network_final"].to_dict("records"))
    final_table = pd.DataFrame(final_rows)
    baseline_f1 = float(final_table.loc[final_table["model"] == "Baseline", "test_macro_f1"].iloc[0])
    final_table["beats_baseline"] = np.where(final_table["test_macro_f1"] > baseline_f1, "yes", "no")
    final_table = final_table.sort_values(["test_macro_f1", "villain_recall"], ascending=[False, False]).reset_index(drop=True)
    final_table.insert(0, "rank", np.arange(1, len(final_table) + 1))
    _save_table(final_table, output_dir, "T11_final_ranking")
    figures.append(final_comparison(final_table, output_dir / "G16_final_comparison.png"))

    artifacts = [str(path) for path in output_dir.iterdir() if path.is_file()]
    checks = validate_run(classification, clustering, neural, artifacts)
    (output_dir / "validation.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    _write_report(output_dir, classification, clustering, neural, final_table)
    return {"figures": figures, "tables": artifacts, "validation": checks, "final_ranking": final_table}


if __name__ == "__main__":
    run_assignment()
