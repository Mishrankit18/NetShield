"""Train and evaluate 8 ML classifiers on Dataset/phishing.csv."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.features import URL_FEATURE_NAMES

matplotlib.use("Agg")

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "Dataset" / "phishing.csv"
MODELS_DIR = ROOT_DIR / "models"
IMAGES_DIR = ROOT_DIR / "Images"


def build_models(random_state: int = 42) -> dict[str, object]:
    """Return the 8 classifiers matching the reference ML-ProjectKart suite."""
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=random_state)
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1),
        "K-Nearest Neighbors": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Support Vector Machine": make_pipeline(StandardScaler(), SVC(probability=True, random_state=random_state)),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
        "AdaBoost": AdaBoostClassifier(random_state=random_state),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def save_confusion_matrix(cm: np.ndarray, model_name: str, out_path: Path) -> None:
    """Render and save a clean confusion matrix figure."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    labels = ["Phishing (0)", "Legitimate (1)"]
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="Actual",
        xlabel="Predicted",
        title=f"{model_name} Confusion Matrix",
    )
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
                fontweight="bold",
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_comparison_chart(results: list[dict], out_path: Path) -> None:
    """Save horizontal bar chart comparing model accuracies."""
    ordered = sorted(results, key=lambda r: r["accuracy"])
    names = [r["model"] for r in ordered]
    scores = [r["accuracy"] * 100 for r in ordered]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.barh(names, scores, color="#2563eb")
    ax.set_xlim(0, 105)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Phishing Website Detection - Model Accuracy Comparison")
    for bar, score in zip(bars, scores):
        ax.text(score + 0.8, bar.get_y() + bar.get_height() / 2, f"{score:.2f}%", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def train_and_evaluate() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    if "Index" in df.columns:
        df = df.drop(columns=["Index"])

    X = df.drop(columns=["class"])
    # Map {-1: 0 (Phishing), 1: 1 (Legitimate)} for uniform binary classification metrics
    y = (df["class"] == 1).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models = build_models(random_state=42)
    results: list[dict] = []
    best_name = ""
    best_acc = -1.0
    best_model = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred)

        slug = name.lower().replace(" ", "_").replace("-", "_")
        save_confusion_matrix(cm, name, IMAGES_DIR / f"cm_{slug}.png")

        results.append(
            {
                "model": name,
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "confusion_matrix": cm.tolist(),
                "image": f"Images/cm_{slug}.png",
            }
        )

        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_model = model

    results.sort(key=lambda r: r["accuracy"], reverse=True)
    save_comparison_chart(results, IMAGES_DIR / "model_comparison.png")

    artifact = {
        "model_name": best_name,
        "model": best_model,
        "feature_names": list(X.columns),
        "label_mapping": {0: "Phishing", 1: "Legitimate"},
    }
    joblib.dump(artifact, MODELS_DIR / "best_model.joblib")

    # Train a dedicated URL-only classifier on the 10 URL-observable features
    url_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    url_model.fit(X_train[URL_FEATURE_NAMES], y_train)
    url_acc = float(accuracy_score(y_test, url_model.predict(X_test[URL_FEATURE_NAMES])))
    url_artifact = {
        "model_name": "Random Forest (URL-only)",
        "model": url_model,
        "feature_names": URL_FEATURE_NAMES,
        "accuracy": round(url_acc, 4),
        "label_mapping": {0: "Phishing", 1: "Legitimate"},
    }
    joblib.dump(url_artifact, MODELS_DIR / "url_model.joblib")

    summary = {
        "dataset_rows": int(len(df)),
        "feature_count": int(X.shape[1]),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "best_model": best_name,
        "best_accuracy": round(best_acc, 4),
        "url_model_accuracy": round(url_acc, 4),
        "results": results,
    }
    (MODELS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Trained {len(results)} models on {len(X_train)} train / {len(X_test)} test samples.")
    print(f"Best model: {best_name} ({best_acc * 100:.2f}% accuracy)")
    for row in results:
        print(
            f"  {row['model']:<24} | Acc: {row['accuracy']*100:6.2f}% | "
            f"F1: {row['f1_score']*100:6.2f}%"
        )


if __name__ == "__main__":
    train_and_evaluate()

