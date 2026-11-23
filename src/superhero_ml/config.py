from __future__ import annotations

SEED = 42
TARGET = "alignment"
NAME_COLUMN = "name"

FEATURES = [
    "intelligence_score",
    "strength_score",
    "speed_score",
    "durability_score",
    "power_score",
    "combat_score",
]

LABEL_MAP = {"Good": 0, "Bad": 1}
CLASS_LABELS = [0, 1]
CLASS_NAMES = ["Hero", "Villain"]

COLORS = {
    "Hero": "#1F5FBF",
    "Villain": "#D62839",
    "train.csv": "#4C78A8",
    "test.csv": "#F58518",
    "Baseline": "#9E9E9E",
    "Decision Tree": "#2E7D32",
    "KNN": "#F9A825",
    "Network A": "#6A1B9A",
    "Network B": "#00838F",
    "Training metric": "#2E7D32",
    "Validation metric": "#F9A825",
    "Cut line": "#333333",
    "Connection": "#B0B0B0",
}

# Cluster colours are deliberately independent of the unsupervised algorithm.
# That keeps Cluster 1/Cluster 2 visually consistent across G8-G11 and T8.
CLUSTER_COLORS = {
    1: "#4C78A8",
    2: "#F58518",
    3: "#54A24B",
    4: "#E45756",
    5: "#72B7B2",
    6: "#B279A2",
    7: "#FF9DA6",
    8: "#9D755D",
}

PROHIBITED_FEATURES = {
    "overall_score",
    "history_text",
    "powers_text",
    "superpowers",
    "creator",
    "gender",
    "height",
    "weight",
    "colours",
    "name",
}

