"""
Universalios treniravimo/validavimo pagalbinės funkcijos klasifikavimo eksperimentams.
Naudojimas:
    from evaluation import holdout_evaluate, crossval_evaluate

    holdout_results, (X_tr, X_val, y_tr, y_val) = holdout_evaluate(models, X, y)
    cv_results = crossval_evaluate(models, X, y, cv=5)
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, List, Optional

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split, StratifiedKFold


def _compute_classification_metrics(
    y_true,
    y_pred,
    average: str = "weighted",
) -> Dict[str, Any]:
    """Pagalbinė funkcija bendroms klasifikavimo metrikoms suskaičiuoti."""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def holdout_evaluate(
    models: Dict[str, BaseEstimator],
    X,
    y,
    val_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
    average: str = "weighted",
    verbose: bool = True,
) -> Tuple[Dict[str, Dict[str, Any]], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Atlieka train/validation (hold-out) vertinimą.
    Grąžina metrikas bei pačias padalintas aibes, kad jas būtų galima naudoti tolesniam mokymui.
    """
    if verbose:
        print("\n==========================")
        print("TRAIN + VALIDATION HOLDOUT")
        print("==========================")

    stratify_arg = y if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=val_size,
        random_state=random_state,
        stratify=stratify_arg,
    )

    if verbose:
        print(f"Train dydis      : {len(X_train)}")
        print(f"Validation dydis : {len(X_val)}")

    results: Dict[str, Dict[str, Any]] = {}

    for name, model in models.items():
        if verbose:
            print(f"\nModelis: {name}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        metrics = _compute_classification_metrics(y_val, y_pred, average=average)
        results[name] = metrics

        if verbose:
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1 ({average}): {metrics['f1']:.4f}")

    if verbose:
        print("\nTRAIN + VALIDATION HOLDOUT COMPLETED\n")

    return results, (X_train, X_val, y_train, y_val)


def crossval_evaluate(
    models: Dict[str, BaseEstimator],
    X,
    y,
    cv: int = 5,
    random_state: int = 42,
    average: str = "weighted",
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Atlieka k-fold kryžminę validaciją (tik train/validation stadija).
    Kiekviename folde modelis apmokomas ir įvertinamas, o gautos metrikos suvidurkinamos.
    """
    if verbose:
        print("\n===========================")
        print("K-FOLD CROSS-VALIDATION")
        print("===========================\n")

    skf = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state,
    )

    results: Dict[str, Dict[str, Any]] = {}

    for name, model in models.items():
        if verbose:
            print(f"\nModelis: {name}")

        fold_metrics: List[Dict[str, Any]] = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            if verbose:
                print(
                    f"  Fold {fold_idx}/{cv} -> train={len(train_idx)}, val={len(val_idx)}"
                )

            model_copy = clone(model)
            model_copy.fit(X_train, y_train)
            y_pred = model_copy.predict(X_val)

            fold_metrics.append(
                _compute_classification_metrics(y_val, y_pred, average=average)
            )

        # Agreguojame metrikas (vidurkis + std)
        aggregated = {}
        for key in ["accuracy", "precision", "recall", "f1"]:
            values = [fold[key] for fold in fold_metrics]
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))

        aggregated["fold_details"] = fold_metrics  # jei reikia išsamių rezultatų
        results[name] = aggregated

        if verbose:
            print(f"  Mean accuracy: {aggregated['accuracy_mean']:.4f}")
            print(f"  Mean F1 ({average}): {aggregated['f1_mean']:.4f}")

    if verbose:
        print("\nCROSS-VALIDATION COMPLETED\n")

    return results