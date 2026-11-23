from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from .config import FEATURES, LABEL_MAP, SEED
from .data import DatasetBundle
from .metrics import classification_metrics


def _classification_evaluation(model, X_train, y_train, X_test, y_test) -> dict[str, Any]:
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_prob = model.predict_proba(X_train)[:, 1] if hasattr(model, "predict_proba") else None
    test_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    return {
        "train": classification_metrics(y_train, train_pred, train_prob),
        "test": classification_metrics(y_test, test_pred, test_prob),
        "train_predictions": train_pred.tolist(),
        "test_predictions": test_pred.tolist(),
        "test_probabilities": None if test_prob is None else test_prob.tolist(),
    }


def run_classification(bundle: DatasetBundle) -> dict[str, Any]:
    train = bundle.train_clean_b
    test = bundle.test_clean_b
    X_train, y_train = train[FEATURES], train["target"]
    X_test, y_test = test[FEATURES], test["target"]

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    tree_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeClassifier(criterion="gini", random_state=SEED)),
    ])
    tree_grid = GridSearchCV(
        tree_pipeline,
        {
            "model__max_depth": [3, 5, 7, 10, None],
            "model__min_samples_leaf": [1, 5, 10],
        },
        cv=cv,
        scoring="f1_macro",
        refit=True,
    )
    tree_grid.fit(X_train, y_train)

    knn_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(metric="euclidean")),
    ])
    knn_grid = GridSearchCV(
        knn_pipeline,
        {
            "model__n_neighbors": [3, 5, 7, 9, 11, 15],
            "model__weights": ["uniform", "distance"],
        },
        cv=cv,
        scoring="f1_macro",
        refit=True,
    )
    knn_grid.fit(X_train, y_train)

    models = {
        "Baseline": baseline,
        "Decision Tree": tree_grid.best_estimator_,
        "KNN": knn_grid.best_estimator_,
    }
    evaluations = {
        name: _classification_evaluation(model, X_train, y_train, X_test, y_test)
        for name, model in models.items()
    }
    return {
        "models": models,
        "evaluations": evaluations,
        "best_params": {
            "Decision Tree": tree_grid.best_params_,
            "KNN": knn_grid.best_params_,
        },
        "best_cv_macro_f1": {
            "Decision Tree": float(tree_grid.best_score_),
            "KNN": float(knn_grid.best_score_),
        },
        "grid_sizes": {"Decision Tree": len(tree_grid.cv_results_["params"]), "KNN": len(knn_grid.cv_results_["params"])},
    }


def _renumber_clusters(labels: np.ndarray, original_scores: pd.DataFrame) -> tuple[np.ndarray, dict[int, int], pd.DataFrame]:
    raw_profiles = original_scores.copy()
    raw_profiles["raw_cluster"] = labels
    means = raw_profiles.groupby("raw_cluster")[FEATURES].mean()
    means["average_score"] = means[FEATURES].mean(axis=1)
    ordered = means.sort_values("average_score").index.tolist()
    mapping = {int(raw_id): i + 1 for i, raw_id in enumerate(ordered)}
    renamed = np.asarray([mapping[int(label)] for label in labels], dtype=int)
    profile = raw_profiles.assign(cluster=renamed).groupby("cluster")[FEATURES].agg(["mean"])
    profile.columns = [column for column, _ in profile.columns]
    profile.insert(0, "size", pd.Series(renamed).value_counts().sort_index())
    profile["average_score"] = profile[FEATURES].mean(axis=1)
    return renamed, mapping, profile.reset_index()


def run_clustering(bundle: DatasetBundle) -> dict[str, Any]:
    clean_a = bundle.train_clean_a
    original_scores = clean_a[FEATURES].copy()
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    scaled = preprocessor.fit_transform(original_scores)

    sweep: list[dict[str, float | int]] = []
    kmeans_by_k: dict[int, KMeans] = {}
    for k in range(2, 9):
        model = KMeans(n_clusters=k, n_init=10, random_state=SEED)
        labels = model.fit_predict(scaled)
        kmeans_by_k[k] = model
        sweep.append({
            "k": k,
            "inertia": float(model.inertia_),
            "silhouette": float(silhouette_score(scaled, labels)),
        })
    best_k = min(sweep, key=lambda row: (-round(float(row["silhouette"]), 3), int(row["k"]))) ["k"]
    kmeans = kmeans_by_k[int(best_k)]
    kmeans_labels, kmeans_mapping, kmeans_profiles = _renumber_clusters(kmeans.labels_, original_scores)

    agglo = AgglomerativeClustering(n_clusters=int(best_k), linkage="ward")
    agglo_raw_labels = agglo.fit_predict(scaled)
    agglo_labels, agglo_mapping, agglo_profiles = _renumber_clusters(agglo_raw_labels, original_scores)

    kmeans_scores = {
        "silhouette": float(silhouette_score(scaled, kmeans_labels)),
        "davies_bouldin": float(davies_bouldin_score(scaled, kmeans_labels)),
        "calinski_harabasz": float(calinski_harabasz_score(scaled, kmeans_labels)),
    }
    agglo_scores = {
        "silhouette": float(silhouette_score(scaled, agglo_labels)),
        "davies_bouldin": float(davies_bouldin_score(scaled, agglo_labels)),
        "calinski_harabasz": float(calinski_harabasz_score(scaled, agglo_labels)),
    }
    pca = PCA(n_components=2, random_state=SEED)
    projection = pca.fit_transform(scaled)
    return {
        "preprocessor": preprocessor,
        "scaled": scaled,
        "sweep": pd.DataFrame(sweep),
        "k": int(best_k),
        "kmeans": kmeans,
        "agglomerative": agglo,
        "kmeans_labels": kmeans_labels,
        "agglomerative_labels": agglo_labels,
        "kmeans_mapping": kmeans_mapping,
        "agglomerative_mapping": agglo_mapping,
        "kmeans_profiles": kmeans_profiles,
        "agglomerative_profiles": agglo_profiles,
        "kmeans_scores": kmeans_scores,
        "agglomerative_scores": agglo_scores,
        "ari": float(adjusted_rand_score(kmeans_labels, agglo_labels)),
        "pca": pca,
        "projection": projection,
        "linkage_matrix": linkage(scaled, method="ward"),
        "alignment": clean_a["alignment"].fillna("").replace({"Neutral": "Neutral"}).to_numpy(),
    }


def _build_network(deep: bool):
    import tensorflow as tf

    layers = [tf.keras.layers.Input(shape=(len(FEATURES),)), tf.keras.layers.Dense(16 if deep else 8, activation="relu")]
    if deep:
        layers.append(tf.keras.layers.Dense(8, activation="relu"))
    layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))
    model = tf.keras.Sequential(layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _evaluate_network(model, X, y) -> dict[str, Any]:
    probability = model.predict(X, verbose=0).ravel()
    prediction = (probability >= 0.5).astype(int)
    return classification_metrics(y, prediction, probability)


def run_neural_networks(bundle: DatasetBundle) -> dict[str, Any]:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(SEED)
    train = bundle.train_clean_b
    test = bundle.test_clean_b
    X = train[FEATURES]
    y = train["target"].to_numpy()
    X_test = test[FEATURES]
    y_test = test["target"].to_numpy()
    X_fit, X_val, y_fit, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_fit_scaled = scaler.fit_transform(imputer.fit_transform(X_fit))
    X_val_scaled = scaler.transform(imputer.transform(X_val))
    X_test_scaled = scaler.transform(imputer.transform(X_test))

    results: dict[str, Any] = {
        "imputer": imputer,
        "scaler": scaler,
        "split_sizes": {"fit": len(X_fit), "validation": len(X_val), "test": len(X_test)},
        "networks": {},
    }
    for name, deep in (("Network A", False), ("Network B", True)):
        tf.keras.utils.set_random_seed(SEED)
        model = _build_network(deep)
        callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
        history = model.fit(
            X_fit_scaled,
            y_fit,
            validation_data=(X_val_scaled, y_val),
            epochs=200,
            batch_size=32,
            callbacks=[callback],
            verbose=0,
        )
        best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
        results["networks"][name] = {
            "model": model,
            "history": history.history,
            "best_epoch": best_epoch,
            "parameters": int(model.count_params()),
            "metrics": {
                "train": _evaluate_network(model, X_fit_scaled, y_fit),
                "validation": _evaluate_network(model, X_val_scaled, y_val),
                "test": _evaluate_network(model, X_test_scaled, y_test),
            },
        }
    return results

