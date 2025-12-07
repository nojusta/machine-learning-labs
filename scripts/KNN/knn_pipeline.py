"""
Complete KNN pipeline for binary classification (classes 4 vs 5).

Parts:
- Validation grid search on train_val data (full features and t-SNE 2D).
- Cross-validation on train_val using the best hyperparameters.
- Final one-time evaluation on test data.
- Plots and CSV/JSON artifacts for reporting.

Rules respected:
- Test sets are never used for tuning.
- Train/validation split is done only inside the train_val set.
- Features are already normalized; no extra scaling is applied.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Headless, writable Matplotlib config
mpl_cache = Path(__file__).resolve().parent / ".mpl-cache"
mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(mpl_cache))
import matplotlib  # noqa: E402

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import ListedColormap
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
    k_values: List[int] = field(
        default_factory=lambda: sorted(set(list(range(1, 22, 2)) + [20]))
    )  # 1..21 odd + 20 to cover k>=20
    cv_splits: int = 5
    pos_label: int = 5
    distances: List[str] = field(default_factory=lambda: ["euclidean", "manhattan"])


@dataclass
class DatasetSpec:
    name: str
    train_val_path: Path
    test_path: Path
    is_2d: bool = False  # for optional decision boundary plots


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.rename(columns={"NObeyesdad": "label"})


def compute_basic_metrics(
    y_true: pd.Series, y_pred: np.ndarray, pos_label: int
) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, pos_label=pos_label, average="binary", zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, pos_label=pos_label, average="binary", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=pos_label, average="binary", zero_division=0)),
    }


def evaluate_grid(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: RunConfig,
    distance: str,
) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []
    for k in config.k_values:
        model = KNeighborsClassifier(n_neighbors=k, metric=distance)
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        metrics = compute_basic_metrics(y_val, y_val_pred, pos_label=config.pos_label)
        metrics.update({"k": k, "distance": distance})
        results.append(metrics)
    return results


def select_best_row(rows: List[Dict[str, float]]) -> Dict[str, float]:
    # Highest F1, then smaller k, then prefer euclidean over manhattan implicitly via distance order
    distance_order = {"euclidean": 0, "manhattan": 1}
    return max(
        rows,
        key=lambda r: (r["f1"], -r["k"], -distance_order.get(r.get("distance", ""), 99)),
    )


def plot_metric_curves(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    metrics = ["accuracy", "precision", "recall", "f1"]
    df_melt = df.melt(
        id_vars=["k", "distance"], value_vars=metrics, var_name="metric", value_name="score"
    )
    sns.lineplot(
        data=df_melt,
        x="k",
        y="score",
        hue="metric",
        style="distance",
        marker="o",
    )
    plt.ylim(0, 1.05)
    plt.xlabel("k (neighbors)")
    plt.ylabel("Score")
    plt.title("Validation metrics vs k")
    plt.grid(True, alpha=0.3)
    plt.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, labels: List[int], title: str, path: Path) -> None:
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
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_roc(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, path: Path, title: str) -> None:
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", color="#1f77b4", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def decision_boundary_2d(
    model: KNeighborsClassifier,
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_plot: pd.DataFrame,
    y_plot: pd.Series,
    path: Path,
    title: str,
) -> None:
    """Decision boundary plot: background fit on X_fit/y_fit, scatter from X_plot/y_plot."""
    assert X_fit.shape[1] == 2 and X_plot.shape[1] == 2, "Decision boundary is for 2D features."
    model.fit(X_fit, y_fit)

    all_points = pd.concat([X_fit, X_plot], axis=0)
    x_range = all_points.iloc[:, 0].max() - all_points.iloc[:, 0].min()
    y_range = all_points.iloc[:, 1].max() - all_points.iloc[:, 1].min()
    padding_x = max(0.05, 0.15 * x_range)
    padding_y = max(0.05, 0.15 * y_range)
    x_min, x_max = all_points.iloc[:, 0].min() - padding_x, all_points.iloc[:, 0].max() + padding_x
    y_min, y_max = all_points.iloc[:, 1].min() - padding_y, all_points.iloc[:, 1].max() + padding_y
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400),
    )
    grid = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=X_fit.columns)
    zz = model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(6.8, 5.6))
    plt.contourf(xx, yy, zz, alpha=0.3, cmap="coolwarm")
    scatter = plt.scatter(
        X_plot.iloc[:, 0],
        X_plot.iloc[:, 1],
        c=y_plot,
        cmap="coolwarm",
        edgecolor="k",
        s=40,
        alpha=0.8,
    )
    plt.title(title)
    plt.xlabel(X_fit.columns[0])
    plt.ylabel(X_fit.columns[1])
    plt.legend(*scatter.legend_elements(), title="Class", loc="upper right")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def evaluate_split(
    model: KNeighborsClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    pos_label: int,
    cm_path: Path,
    cm_title: str,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_eval)
    metrics = compute_basic_metrics(y_eval, y_pred, pos_label=pos_label)
    cm = confusion_matrix(y_eval, y_pred, labels=[4, 5])
    plot_confusion_matrix(cm, labels=[4, 5], title=cm_title, path=cm_path)
    proba_index = list(model.classes_).index(pos_label)
    y_proba = model.predict_proba(X_eval)[:, proba_index]
    return metrics, y_pred, y_proba


def evaluate_on_test(
    model: KNeighborsClassifier,
    X_train_val: pd.DataFrame,
    y_train_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    plots_dir: Path,
    pos_label: int,
    prefix: str,
) -> Dict[str, object]:
    metrics, y_pred, y_proba = evaluate_split(
        model,
        X_train_val,
        y_train_val,
        X_test,
        y_test,
        pos_label=pos_label,
        cm_path=plots_dir / f"{prefix}_test_confusion_matrix.png",
        cm_title=f"{prefix} Test confusion matrix",
    )

    fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=pos_label)
    roc_auc = float(auc(fpr, tpr))
    roc_path = plots_dir / f"{prefix}_test_roc_curve.png"
    plot_roc(fpr, tpr, roc_auc, roc_path, title=f"{prefix} Test ROC curve")

    return {
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[4, 5]).tolist(),
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": roc_auc,
        },
        "plots": {
            "confusion_matrix": str(plots_dir / f"{prefix}_test_confusion_matrix.png"),
            "roc_curve": str(roc_path),
        },
        "y_pred": y_pred.tolist(),
        "y_proba": y_proba.tolist(),
    }


def run_cross_validation(
    X: pd.DataFrame, y: pd.Series, k: int, distance: str, config: RunConfig
) -> Dict[str, float]:
    skf = StratifiedKFold(
        n_splits=config.cv_splits,
        shuffle=True,
        random_state=config.random_state,
    )
    fold_metrics: List[Dict[str, float]] = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = KNeighborsClassifier(n_neighbors=k, metric=distance)
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
    X_test: pd.DataFrame,
    y_true: pd.Series,
    y_pred: List[int],
    y_proba: List[float],
    pos_label: int,
    save_path: Path,
) -> Dict[str, object]:
    df = X_test.copy()
    df["true_label"] = y_true.values
    df["pred_label"] = y_pred
    df["pred_proba_pos"] = y_proba
    mis = df[df["true_label"] != df["pred_label"]].copy()
    mis["index"] = mis.index

    fp = mis[(mis["true_label"] == 4) & (mis["pred_label"] == pos_label)]
    fn = mis[(mis["true_label"] == pos_label) & (mis["pred_label"] == 4)]

    feature_cols = list(X_test.columns)

    def summarize(subset: pd.DataFrame, reference_label: int) -> Dict[str, object]:
        if subset.empty:
            return {"count": 0, "top_feature_deviations": [], "examples": []}
        ref = df[df["true_label"] == reference_label]
        ref_mean = ref[feature_cols].mean()
        subset_mean = subset[feature_cols].mean()
        deltas = (subset_mean - ref_mean).sort_values(key=lambda s: s.abs(), ascending=False)
        top = [
            {"feature": feat, "shift_vs_class_mean": float(deltas[feat])}
            for feat in deltas.index[:3]
        ]
        examples = subset[feature_cols + ["true_label", "pred_label", "pred_proba_pos", "index"]].head(
            5
        ).to_dict(orient="records")
        return {"count": int(len(subset)), "top_feature_deviations": top, "examples": examples}

    boundary_like = float((mis["pred_proba_pos"].between(0.4, 0.6)).mean()) if len(mis) else 0.0

    save_path.parent.mkdir(parents=True, exist_ok=True)
    mis[["index", "true_label", "pred_label", "pred_proba_pos"]].to_csv(save_path, index=False)

    return {
        "total_misclassified": int(len(mis)),
        "fraction_near_boundary_prob_0.4_0.6": boundary_like,
        "false_positive": summarize(fp, reference_label=4),
        "false_negative": summarize(fn, reference_label=pos_label),
        "csv_path": str(save_path),
    }


def run_experiment(spec: DatasetSpec, config: RunConfig, base_dir: Path) -> Dict[str, object]:
    df_train_val = load_dataset(spec.train_val_path)
    df_test = load_dataset(spec.test_path)
    feature_cols = [c for c in df_train_val.columns if c != "label"]

    out_dir = base_dir / spec.name
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
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

    all_val_rows: List[Dict[str, float]] = []
    for distance in config.distances:
        all_val_rows.extend(evaluate_grid(X_train, y_train, X_val, y_val, config, distance=distance))

    val_df = pd.DataFrame(all_val_rows)
    val_metrics_path = out_dir / "validation_metrics_per_k.csv"
    val_df.to_csv(val_metrics_path, index=False)
    plot_metric_curves(val_df, path=plots_dir / "validation_metrics_vs_k.png")

    best_entry = select_best_row(all_val_rows)
    best_k = int(best_entry["k"])
    best_distance = best_entry["distance"]

    # Validation confusion matrix for the chosen k/distance
    best_val_model = KNeighborsClassifier(n_neighbors=best_k, metric=best_distance)
    val_metrics_best, y_val_pred, y_val_proba = evaluate_split(
        best_val_model,
        X_train,
        y_train,
        X_val,
        y_val,
        pos_label=config.pos_label,
        cm_path=plots_dir / "validation_confusion_matrix.png",
        cm_title=f"{spec.name} Validation confusion matrix (k={best_k}, {best_distance})",
    )

    cv_results = run_cross_validation(X, y, best_k, distance=best_distance, config=config)

    # Final test evaluation (train on full train_val)
    final_model = KNeighborsClassifier(n_neighbors=best_k, metric=best_distance)
    test_eval = evaluate_on_test(
        final_model,
        X_train_val=X,
        y_train_val=y,
        X_test=df_test[feature_cols],
        y_test=df_test["label"],
        plots_dir=plots_dir,
        pos_label=config.pos_label,
        prefix="full" if spec.name == "full_features" else spec.name,
    )

    error_details = analyze_errors(
        df_test[feature_cols],
        df_test["label"],
        test_eval["y_pred"],
        test_eval["y_proba"],
        config.pos_label,
        save_path=out_dir / "misclassified_indices.csv",
    )

    # Decision boundary plots: separate train/val and test views (background fit on train/val)
    boundary_plot_train = None
    boundary_plot_test = None
    if spec.is_2d:
        boundary_plot_train = str(plots_dir / "decision_boundary_tsne_trainval.png")
        boundary_plot_test = str(plots_dir / "decision_boundary_tsne_test.png")
        decision_boundary_2d(
            KNeighborsClassifier(n_neighbors=best_k, metric=best_distance),
            X_fit=X,
            y_fit=y,
            X_plot=X,
            y_plot=y,
            path=plots_dir / "decision_boundary_tsne_trainval.png",
            title=f"{spec.name} decision boundary (train/val, k={best_k}, {best_distance})",
        )
        decision_boundary_2d(
            KNeighborsClassifier(n_neighbors=best_k, metric=best_distance),
            X_fit=X,
            y_fit=y,
            X_plot=df_test[feature_cols],
            y_plot=df_test["label"],
            path=plots_dir / "decision_boundary_tsne_test.png",
            title=f"{spec.name} decision boundary (test, k={best_k}, {best_distance})",
        )
    elif {"FCVC", "FAF"}.issubset(feature_cols):
        two_feats = ["FCVC", "FAF"]
        boundary_plot_train = str(plots_dir / "decision_boundary_fcvc_faf_trainval.png")
        boundary_plot_test = str(plots_dir / "decision_boundary_fcvc_faf_test.png")
        decision_boundary_2d(
            KNeighborsClassifier(n_neighbors=best_k, metric=best_distance),
            X_fit=df_train_val[two_feats],
            y_fit=y,
            X_plot=df_train_val[two_feats],
            y_plot=y,
            path=plots_dir / "decision_boundary_fcvc_faf_trainval.png",
            title=f"Decision boundary on FCVC/FAF (train/val, k={best_k}, {best_distance})",
        )
        decision_boundary_2d(
            KNeighborsClassifier(n_neighbors=best_k, metric=best_distance),
            X_fit=df_train_val[two_feats],
            y_fit=y,
            X_plot=df_test[two_feats],
            y_plot=df_test["label"],
            path=plots_dir / "decision_boundary_fcvc_faf_test.png",
            title=f"Decision boundary on FCVC/FAF (test, k={best_k}, {best_distance})",
        )

    # Drop heavy arrays before saving JSON
    test_eval.pop("y_pred", None)
    test_eval.pop("y_proba", None)

    return {
        "dataset": spec.name,
        "features": feature_cols,
        "data_summary": {
            "train_val_samples": int(len(df_train_val)),
            "test_samples": int(len(df_test)),
            "train_val_class_counts": df_train_val["label"].value_counts().to_dict(),
            "test_class_counts": df_test["label"].value_counts().to_dict(),
            "train_size": int(len(X_train)),
            "val_size": int(len(X_val)),
        },
        "best_hyperparams": {
            "k": best_k,
            "distance": best_distance,
            "validation_metrics": val_metrics_best,
        },
        "validation": {
            "per_k": all_val_rows,
            "best_confusion_matrix_path": str(plots_dir / "validation_confusion_matrix.png"),
            "metrics_csv": str(val_metrics_path),
            "metrics_plot": str(plots_dir / "validation_metrics_vs_k.png"),
        },
        "cross_validation": cv_results,
        "test": test_eval,
        "error_analysis": error_details,
        "artifacts": {
            "plots_dir": str(plots_dir),
            "boundary_plot_train": boundary_plot_train,
            "boundary_plot_test": boundary_plot_test,
        },
    }


def save_json(data: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data"
    artifacts_dir = Path(__file__).resolve().parent / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    config = RunConfig()

    specs = [
        DatasetSpec(
            name="full_features",
            train_val_path=data_dir / "classification_train_val.csv",
            test_path=data_dir / "classification_test.csv",
            is_2d=False,
        ),
        DatasetSpec(
            name="tsne_2d",
            train_val_path=data_dir / "classification_train_val_tsne.csv",
            test_path=data_dir / "classification_test_tsne.csv",
            is_2d=True,
        ),
    ]

    experiments: Dict[str, object] = {}
    for spec in specs:
        experiments[spec.name] = run_experiment(spec, config=config, base_dir=artifacts_dir)
        print(
            f"[{spec.name}] best k={experiments[spec.name]['best_hyperparams']['k']} "
            f"distance={experiments[spec.name]['best_hyperparams']['distance']} "
            f"val F1={experiments[spec.name]['best_hyperparams']['validation_metrics']['f1']:.4f}"
        )

    results = {
        "config": {
            "random_state": config.random_state,
            "val_size": config.val_size,
            "k_values": config.k_values,
            "cv_splits": config.cv_splits,
            "pos_label": config.pos_label,
            "distances": config.distances,
        },
        "experiments": experiments,
    }

    json_path = artifacts_dir / "knn_results.json"
    save_json(results, json_path)
    print(f"\nAll results saved to: {json_path}")


if __name__ == "__main__":
    main()
