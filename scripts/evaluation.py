"""
Bendros vertinimo funkcijos klasifikatoriams (sklearn stiliaus):
- Hold-out strategija (train/test split)
- k-fold kryžminė validacija

Naudojimas:

    from evaluation import holdout_evaluate, crossval_evaluate

    models = {
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    holdout_results, splits = holdout_evaluate(models, X, y)
    cv_results = crossval_evaluate(models, X, y, cv=5)
"""

from typing import Dict, Any, Tuple, Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)


def holdout_evaluate(
    models: Dict[str, BaseEstimator],
    X,
    y,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[Dict[str, Dict[str, Any]], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Hold-out (train/test split) vertinimas BENDRAI keliems modeliams.

    :param models: dict {"pavadinimas": NEAPMOKYTAS modelio_objektas}
                   Gali būti bet kuris sklearn klasifikatorius arba Pipeline.
    :param X: požymių matrica (numpy array, pandas DataFrame ir pan.)
    :param y: klasių vektorius (numpy array, pandas Series)
    :param test_size: testinės aibės dalis (0.2 = 20%)
    :param random_state: kad padalinimas būtų pakartojamas
    :param stratify: ar naudoti stratifikaciją pagal klases
    :return: (rezultatai_žodynas, (X_train, X_test, y_train, y_test))

    results struktūra:
        {
          "modelio_vardas": {
              "accuracy": ...,
              "precision_weighted": ...,
              "recall_weighted": ...,
              "f1_weighted": ...,
              "classification_report": string
          },
          ...
        }
    """

    print("\n====================")
    print("HOLD-OUT EVALUATION")
    print("====================")

    print(f"\n1. Duomenų padalinimas į train/test su test_size={test_size} ir random_state={random_state}")
    stratify_arg = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg,
    )

    print(f"   - Train dydis: {len(X_train)}")
    print(f"   - Test dydis : {len(X_test)}")

    results: Dict[str, Dict[str, Any]] = {}

    for name, model in models.items():
        print(f"\n2. Modelio treniravimas: {name}")
        model.fit(X_train, y_train)

        print("3. Prognozavimas ant test rinkinio")
        y_pred = model.predict(X_test)

        print("4. Metrikų skaičiavimas")
        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )

        results[name] = {
            "accuracy": acc,
            "precision_weighted": precision,
            "recall_weighted": recall,
            "f1_weighted": f1,
            "classification_report": classification_report(
                y_test, y_pred, zero_division=0
            ),
        }

        print(f"   - Accuracy: {acc:.4f}")
        print(f"   - Weighted F1: {f1:.4f}")

    print("\nHOLD-OUT EVALUATION COMPLETED\n")

    return results, (X_train, X_test, y_train, y_test)


def crossval_evaluate(
    models: Dict[str, BaseEstimator],
    X,
    y,
    cv: int = 5,
    random_state: int = 42,
    scoring: Optional[dict] = None,
) -> Dict[str, Dict[str, float]]:
    """
    k-fold kryžminė validacija BENDRAI keliems modeliams.

    :param models: dict {"pavadinimas": NEAPMOKYTAS modelio_objektas}
    :param X: požymių matrica
    :param y: klasių vektorius
    :param cv: fold'ų skaičius (k)
    :param random_state: StratifiedKFold
    :param scoring: sklearn scoring dict, pvz.:
        {
            "accuracy": "accuracy",
            "f1_weighted": "f1_weighted",
        }
      jei None – naudos tik accuracy
    :return: dict:
        {
          "modelio_vardas": {
              "accuracy_mean": ...,
              "accuracy_std": ...,
              "f1_weighted_mean": ...,
              "f1_weighted_std": ...,
              ...
          },
          ...
        }
    """

    print("\n===========================")
    print("K-FOLD CROSS-VALIDATION")
    print("===========================\n")

    if scoring is None:
        scoring = {"accuracy": "accuracy"}

    print(f"1. Kuriamas StratifiedKFold su k={cv}, shuffle=True ir random_state={random_state}")
    skf = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state,
    )

    print(f"\n2. Naudojamos metrikos: {list(scoring.keys())}")

    results: Dict[str, Dict[str, float]] = {}

    for name, model in models.items():
        print(f"\n--- Modelio '{name}' vertinimas per {cv} fold'us ---")

        cv_res = cross_validate(
            model,
            X,
            y,
            cv=skf,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        model_result: Dict[str, float] = {}

        for key, values in cv_res.items():
            if key.startswith("test_"):
                metric_name = key.replace("test_", "")
                mean_val = float(np.mean(values))
                std_val = float(np.std(values))
                model_result[metric_name + "_mean"] = mean_val
                model_result[metric_name + "_std"] = std_val

                print(f"   * {metric_name}: mean={mean_val:.4f}, std={std_val:.4f}")

        results[name] = model_result

    print("\nCROSS-VALIDATION COMPLETED\n")

    return results