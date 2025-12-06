"""
KNN training/validation/test pipeline for binary classification (obesity classes 4 vs 5).

Pipeline:
- Train/validation split on classification_train_val.csv to pick best k (odd numbers).
- Euclidean distance, best k chosen by validation F1 (positive class = 5).
- Cross-validation on train+validation data with the chosen k.
- Final evaluation on classification_test.csv.
- Metrics, plots, and JSON outputs are written to scripts/KNN/artifacts/.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
mpl_cache = Path(__file__).resolve().parent / ".mpl-cache"
mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(mpl_cache))
import matplotlib  # noqa: E402

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier


@dataclass
class RunConfig:
    random_state: int = 42
    val_size: float = 0.2
    k_values: List[int] = field(default_factory=lambda: list(range(1, 22, 2)))
    cv_splits: int = 5
    pos_label: int = 5


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Load train/validation and test CSVs and align the target column name."""
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data"
    train_val_path = data_dir / "classification_train_val.csv"
    test_path = data_dir / "classification_test.csv"

    df_train_val = pd.read_csv(train_val_path)
    df_test = pd.read_csv(test_path)

    df_train_val = df_train_val.rename(columns={"NObeyesdad": "label"})
    df_test = df_test.rename(columns={"NObeyesdad": "label"})

    feature_cols = [c for c in df_train_val.columns if c != "label"]
    return df_train_val, df_test, feature_cols


def compute_basic_metrics(
    y_true: pd.Series, y_pred: np.ndarray, pos_label: int
) -> Dict[str, float]:
    """Return accuracy/precision/recall/F1 with a fixed positive class."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, pos_label=pos_label, average="binary", zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, pos_label=pos_label, average="binary", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=pos_label, average="binary", zero_division=0)),
    }


def evaluate_k_grid(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: RunConfig,
) -> List[Dict[str, float]]:
    """Train/evaluate a range of k values on the holdout validation set."""
    results: List[Dict[str, float]] = []

    for k in config.k_values:
        model = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        metrics = compute_basic_metrics(y_val, y_val_pred, pos_label=config.pos_label)
        metrics["k"] = k
        results.append(metrics)

    return results


def select_best_k(validation_results: List[Dict[str, float]]) -> Dict[str, float]:
    """Pick the entry with the highest F1; ties resolved by smaller k."""
    return max(validation_results, key=lambda row: (row["f1"], -row["k"]))


def plot_metric_curves(
    validation_results: List[Dict[str, float]], plots_dir: Path
) -> Path:
    """Line plot for validation metrics over k."""
    df = pd.DataFrame(validation_results)
    plt.figure(figsize=(8, 5))
    for metric, label in [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1 score"),
    ]:
        sns.lineplot(data=df, x="k", y=metric, marker="o", label=label)

    plt.xlabel("k (number of neighbors)")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.title("Validation metrics vs k")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plots_dir.mkdir(parents=True, exist_ok=True)
    path = plots_dir / "validation_metrics_vs_k.png"
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_confusion_matrix(cm: np.ndarray, labels: List[int], path: Path) -> None:
    """Save a labeled confusion matrix heatmap."""
    plt.figure(figsize=(4, 3.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Test confusion matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_roc(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, path: Path) -> None:
    """Save ROC curve plot."""
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", color="#1f77b4", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Test ROC curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def evaluate_on_test(
    model: KNeighborsClassifier,
    X_train_val: pd.DataFrame,
    y_train_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    plots_dir: Path,
    pos_label: int,
) -> Dict[str, object]:
    """Train on full train+val data, evaluate on test, and save plots."""
    model.fit(X_train_val, y_train_val)
    y_pred = model.predict(X_test)
    metrics = compute_basic_metrics(y_test, y_pred, pos_label=pos_label)

    cm = confusion_matrix(y_test, y_pred, labels=[4, 5])
    cm_path = plots_dir / "test_confusion_matrix.png"
    plot_confusion_matrix(cm, labels=[4, 5], path=cm_path)

    proba_index = list(model.classes_).index(pos_label)
    y_proba = model.predict_proba(X_test)[:, proba_index]
    fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=pos_label)
    roc_auc = float(auc(fpr, tpr))
    roc_path = plots_dir / "test_roc_curve.png"
    plot_roc(fpr, tpr, roc_auc, roc_path)

    return {
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": roc_auc,
        },
        "plots": {
            "confusion_matrix": str(cm_path),
            "roc_curve": str(roc_path),
        },
        "y_pred": y_pred.tolist(),
    }


def run_cross_validation(
    X: pd.DataFrame, y: pd.Series, k: int, config: RunConfig
) -> Dict[str, float]:
    """Stratified k-fold cross-validation on train+validation data."""
    skf = StratifiedKFold(
        n_splits=config.cv_splits,
        shuffle=True,
        random_state=config.random_state,
    )
    fold_metrics: List[Dict[str, float]] = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        fold_metrics.append(compute_basic_metrics(y_val, y_pred, pos_label=config.pos_label))

    aggregated = {}
    for metric in ["accuracy", "precision", "recall", "f1"]:
        values = [fold[metric] for fold in fold_metrics]
        aggregated[f"{metric}_mean"] = float(np.mean(values))
        aggregated[f"{metric}_std"] = float(np.std(values))

    return aggregated


def analyze_errors(
    X_test: pd.DataFrame, y_true: pd.Series, y_pred: List[int], pos_label: int
) -> Dict[str, object]:
    """Inspect false positives/negatives and their feature tendencies."""
    annotated = X_test.copy()
    annotated["true_label"] = y_true.values
    annotated["pred_label"] = y_pred

    fp = annotated[(annotated["true_label"] == 4) & (annotated["pred_label"] == pos_label)]
    fn = annotated[(annotated["true_label"] == pos_label) & (annotated["pred_label"] == 4)]

    feature_cols = list(X_test.columns)

    def summarize(subset: pd.DataFrame, reference_label: int) -> Dict[str, object]:
        if subset.empty:
            return {"count": 0, "top_feature_deviations": [], "examples": []}

        ref = annotated[annotated["true_label"] == reference_label]
        ref_mean = ref[feature_cols].mean()
        subset_mean = subset[feature_cols].mean()
        deltas = (subset_mean - ref_mean).sort_values(key=lambda s: s.abs(), ascending=False)
        top = [
            {"feature": feat, "shift_vs_class_mean": float(deltas[feat])}
            for feat in deltas.index[:3]
        ]

        examples = (
            subset[feature_cols + ["true_label", "pred_label"]]
            .head(5)
            .to_dict(orient="records")
        )

        return {
            "count": int(len(subset)),
            "top_feature_deviations": top,
            "examples": examples,
        }

    return {
        "false_positive": summarize(fp, reference_label=4),
        "false_negative": summarize(fn, reference_label=pos_label),
    }


def save_json(data: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    config = RunConfig()
    df_train_val, df_test, feature_cols = load_datasets()

    artifacts_dir = Path(__file__).resolve().parent / "artifacts"
    plots_dir = artifacts_dir / "plots"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    X = df_train_val[feature_cols]
    y = df_train_val["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=config.val_size,
        stratify=y,
        random_state=config.random_state,
    )

    validation_results = evaluate_k_grid(X_train, y_train, X_val, y_val, config=config)
    best_entry = select_best_k(validation_results)
    best_k = int(best_entry["k"])

    val_metrics_path = artifacts_dir / "validation_metrics_per_k.csv"
    pd.DataFrame(validation_results).to_csv(val_metrics_path, index=False)
    metrics_plot_path = plot_metric_curves(validation_results, plots_dir=plots_dir)

    best_model = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean")
    test_eval = evaluate_on_test(
        best_model,
        X_train_val=X,
        y_train_val=y,
        X_test=df_test[feature_cols],
        y_test=df_test["label"],
        plots_dir=plots_dir,
        pos_label=config.pos_label,
    )
    predictions = test_eval.pop("y_pred")

    cv_results = run_cross_validation(X, y, best_k, config=config)
    error_details = analyze_errors(df_test[feature_cols], df_test["label"], predictions, config.pos_label)

    results = {
        "config": {
            "random_state": config.random_state,
            "val_size": config.val_size,
            "k_grid": config.k_values,
            "cv_splits": config.cv_splits,
            "pos_label": config.pos_label,
        },
        "data_summary": {
            "train_val_samples": int(len(df_train_val)),
            "test_samples": int(len(df_test)),
            "train_val_class_counts": df_train_val["label"].value_counts().to_dict(),
            "test_class_counts": df_test["label"].value_counts().to_dict(),
            "train_size": int(len(X_train)),
            "val_size": int(len(X_val)),
        },
        "validation": {
            "best_k": best_k,
            "best_metrics": {k: float(v) for k, v in best_entry.items() if k != "k"},
            "per_k": validation_results,
            "artifacts": {
                "metrics_csv": str(val_metrics_path),
                "metrics_plot": str(metrics_plot_path),
            },
        },
        "cross_validation": cv_results,
        "test": test_eval,
        "error_analysis": error_details,
        "artifacts": {
            "plots_dir": str(plots_dir),
            "json_path": str(artifacts_dir / "knn_results.json"),
        },
    }

    json_path = artifacts_dir / "knn_results.json"
    save_json(results, json_path)

    print(f"\nBest k (by validation F1): {best_k}")
    print(f"Validation F1: {best_entry['f1']:.4f}")
    print(f"Test accuracy: {test_eval['metrics']['accuracy']:.4f}")
    print(f"Test F1: {test_eval['metrics']['f1']:.4f}")
    print(f"Cross-val mean F1 ({config.cv_splits} folds): {cv_results['f1_mean']:.4f}")
    print(f"Results saved to: {json_path}")


if __name__ == "__main__":
    main()
