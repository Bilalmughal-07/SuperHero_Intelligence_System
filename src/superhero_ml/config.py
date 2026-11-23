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
    "Baseline": "#9E9E9E",
    "Decision Tree": "#2E7D32",
    "KNN": "#F9A825",
    "Network A": "#6A1B9A",
    "Network B": "#00838F",
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

