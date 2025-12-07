"""
ROC comparison for KNN, Decision Tree, and Random Forest on both original and t-SNE feature spaces.
- Original: classification_train_val.csv -> classification_test.csv
- t-SNE: classification_train_val_tsne.csv -> classification_test_tsne.csv

Outputs:
- scripts/KNN/artifacts/roc_comparison.png        (original features)
- scripts/KNN/artifacts/roc_comparison_tsne.png  (t-SNE 2D)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

# Use a writable cache to avoid font warnings
mpl_cache = Path(__file__).resolve().parent / "KNN" / ".mpl-cache"
mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(mpl_cache))
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import auc, roc_curve  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402


def load_dataset(train_name: str, test_name: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    root = Path(__file__).resolve().parents[1]
    train = pd.read_csv(root / "data" / train_name).rename(columns={"NObeyesdad": "label"})
    test = pd.read_csv(root / "data" / test_name).rename(columns={"NObeyesdad": "label"})
    # Remove exact overlaps to avoid leakage
    train_tuples = train.apply(tuple, axis=1)
    test_set = set(test.apply(tuple, axis=1))
    train = train.loc[~train_tuples.isin(test_set)].reset_index(drop=True)
    feature_cols = [c for c in train.columns if c != "label"]
    return train[feature_cols], train["label"], test[feature_cols], test["label"]


def fit_models(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, object]:
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5, metric="euclidean"),
        "Decision Tree": DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def compute_roc(models, X_test, y_test, pos_label=5):
    roc_data = {}
    for name, model in models.items():
        proba = model.predict_proba(X_test)[:, list(model.classes_).index(pos_label)]
        fpr, tpr, _ = roc_curve(y_test, proba, pos_label=pos_label)
        roc_data[name] = {"fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr)}
    return roc_data


def plot_roc(roc_data, title: str, save_path: Path):
    plt.figure(figsize=(7, 6))
    colors = {"KNN": "#1f77b4", "Random Forest": "#d62728", "Decision Tree": "#2ca02c"}
    for name, data in roc_data.items():
        plt.plot(
            data["fpr"],
            data["tpr"],
            label=f"{name} ROC (AUC = {data['auc']:.2f})",
            linewidth=2,
            color=colors.get(name, None),
        )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Random chance")
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=220)
    plt.close()


def run_for_dataset(train_csv: str, test_csv: str, title: str, save_name: str):
    X_train, y_train, X_test, y_test = load_dataset(train_csv, test_csv)
    models = fit_models(X_train, y_train)
    roc_data = compute_roc(models, X_test, y_test, pos_label=5)
    out_path = Path(__file__).resolve().parents[1] / "KNN" / "artifacts" / save_name
    plot_roc(roc_data, title, out_path)
    for name, data in roc_data.items():
        print(f"{save_name} | {name}: AUC={data['auc']:.4f}")
    print(f"{save_name} saved to: {out_path}")


def main():
    run_for_dataset(
        train_csv="classification_train_val.csv",
        test_csv="classification_test.csv",
        title="ROC curves comparison on test set (original feature space)",
        save_name="roc_comparison.png",
    )
    run_for_dataset(
        train_csv="classification_train_val_tsne.csv",
        test_csv="classification_test_tsne.csv",
        title="ROC curves comparison on test set (t-SNE feature space)",
        save_name="roc_comparison_tsne.png",
    )


if __name__ == "__main__":
    main()
