from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import ConfusionMatrixDisplay

from .config import CLASS_NAMES, COLORS, FEATURES


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def class_counts(train: pd.DataFrame, test: pd.DataFrame, path: Path) -> str:
    labels = ["Good", "Bad"]
    train_counts = train["alignment"].value_counts().reindex(labels, fill_value=0)
    test_counts = test["alignment"].value_counts().reindex(labels, fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(2)
    width = 0.36
    bars_a = ax.bar(x - width / 2, train_counts, width, label="train.csv", color="#5B8FF9")
    bars_b = ax.bar(x + width / 2, test_counts, width, label="test.csv", color="#61DDAA")
    for bars, counts in ((bars_a, train_counts), (bars_b, test_counts)):
        total = counts.sum()
        for bar, count in zip(bars, counts):
            pct = 0 if total == 0 else 100 * count / total
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{pct:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_title("G1 — Good and Bad class counts")
    ax.set_ylabel("Rows")
    ax.set_xticks(x, ["Good", "Bad"])
    ax.legend()
    return _save(fig, path)


def confusion(metrics: dict, title: str, path: Path) -> str:
    matrix = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    display = ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES)
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    return _save(fig, path)


def model_comparison(rows: pd.DataFrame, title: str, path: Path, metrics: list[str]) -> str:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(rows))
    width = 0.8 / len(metrics)
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - (len(metrics) - 1) / 2) * width, rows[metric], width, label=metric.replace("_", " ").title())
    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_xticks(x, rows.index)
    ax.legend()
    return _save(fig, path)


def clustering_sweep(sweep: pd.DataFrame, path_elbow: Path, path_silhouette: Path, final_k: int) -> tuple[str, str]:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sweep["k"], sweep["inertia"], marker="o")
    ax.set_title("G5 — K-Means elbow curve")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia")
    elbow_path = _save(fig, path_elbow)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sweep["k"], sweep["silhouette"], marker="o")
    ax.axvline(final_k, linestyle="--", color="black", label=f"Final k={final_k}")
    ax.set_title("G6 — Silhouette score by k")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.legend()
    silhouette_path = _save(fig, path_silhouette)
    return elbow_path, silhouette_path


def dendrogram_plot(linkage_matrix, final_k: int, path: Path) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(linkage_matrix, truncate_mode="lastp", p=30, ax=ax, color_threshold=None)
    distances = linkage_matrix[:, 2]
    cut_index = max(0, len(distances) - final_k)
    cut_height = float(distances[cut_index]) if len(distances) else 0.0
    ax.axhline(cut_height, linestyle="--", color="black", label=f"Cut for k={final_k}")
    ax.set_title("G7 — Ward dendrogram")
    ax.set_xlabel("Merged groups")
    ax.set_ylabel("Ward distance")
    ax.legend()
    return _save(fig, path)


def pca_scatter(projection: np.ndarray, labels: np.ndarray, explained: np.ndarray, title: str, path: Path) -> str:
    fig, ax = plt.subplots(figsize=(7, 5))
    for cluster in sorted(np.unique(labels)):
        mask = labels == cluster
        ax.scatter(projection[mask, 0], projection[mask, 1], s=24, label=f"Cluster {cluster}")
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")
    ax.legend()
    return _save(fig, path)


def profiles(profile: pd.DataFrame, title: str, path: Path) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(FEATURES))
    width = 0.8 / max(1, len(profile))
    for i, (_, row) in enumerate(profile.set_index("cluster").iterrows()):
        ax.bar(x + (i - (len(profile) - 1) / 2) * width, [row[f] for f in FEATURES], width, label=f"Cluster {int(row.name)}")
    ax.set_title(title)
    ax.set_xlabel("Ability score")
    ax.set_ylabel("Average original-scale score")
    ax.set_xticks(x, FEATURES, rotation=30, ha="right")
    ax.set_ylim(0, 100)
    ax.legend()
    return _save(fig, path)


def architecture(path: Path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, title, layers, color in [
        (axes[0], "Network A", [6, 8, 1], COLORS["Network A"]),
        (axes[1], "Network B", [6, 16, 8, 1], COLORS["Network B"]),
    ]:
        x_positions = np.linspace(0.1, 0.9, len(layers))
        for x, count in zip(x_positions, layers):
            ys = np.linspace(0.15, 0.85, min(count, 10))
            ax.scatter([x] * len(ys), ys, color=color, s=40)
            ax.text(x, 0.05, str(count), ha="center")
        for left, right in zip(x_positions[:-1], x_positions[1:]):
            ax.plot([left, right], [0.5, 0.5], color="#BDBDBD", linewidth=0.6)
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
    fig.suptitle("G12 — Neural-network architectures")
    return _save(fig, path)


def loss_curve(history: dict, title: str, best_epoch: int, path: Path) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = np.arange(1, len(history["loss"]) + 1)
    ax.plot(epochs, history["loss"], label="Training loss")
    ax.plot(epochs, history["val_loss"], label="Validation loss")
    ax.axvline(best_epoch, linestyle="--", color="black", label=f"Best epoch={best_epoch}")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary cross-entropy loss")
    ax.legend()
    return _save(fig, path)


def final_comparison(rows: pd.DataFrame, path: Path) -> str:
    return model_comparison(rows.set_index("model"), "G16 — Final model comparison", path, ["test_accuracy", "test_macro_f1"])

