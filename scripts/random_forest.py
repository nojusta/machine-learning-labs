import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from evaluation import holdout_evaluate, crossval_evaluate


TARGET_COL = "NObeyesdad"


def print_test_metrics(title: str, y_true, y_pred) -> None:
    """Pagalbinė funkcija galutinio TEST rinkinio metrikoms atspausdinti."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    print(f"Accuracy       : {acc:.4f}")
    print(f"Precision (w)  : {precision:.4f}")
    print(f"Recall (w)     : {recall:.4f}")
    print(f"F1 (w)         : {f1:.4f}")
    print("\nConfusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, zero_division=0))


def get_rf_configurations():
    """
    Keli skirtingi Random Forest konfigūracijų variantai,
    kad būtų galima palyginti jų veikimą.
    """
    configs = {
        "rf_baseline": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
        "rf_more_trees": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),
        "rf_depth_5": RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            random_state=42,
            n_jobs=-1,
        ),
        "rf_depth_10": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        ),
    }
    return configs


def experiment_original_features():
    """
    Eksperimentai su pradiniu 6 požymių rinkiniu (classification_train_val.csv / classification_test.csv)
    1) Hold-out train/validation
    2) k-fold kryžminė validacija
    3) Galutinis testavimas ant classification_test.csv
    """
    print("\n" + "#" * 60)
    print("RANDOM FOREST – ORIGINALŪS 6 POŽYMIAI")
    print("#" * 60)

    # --- Duomenų užkrovimas ---
    df_train_val = pd.read_csv("./data/classification_train_val.csv")
    df_test = pd.read_csv("./data/classification_test.csv")

    X_train_val = df_train_val.drop(columns=[TARGET_COL])
    y_train_val = df_train_val[TARGET_COL]

    X_test = df_test.drop(columns=[TARGET_COL])
    y_test = df_test[TARGET_COL]

    # --- Modelių rinkinys (skirtingi hiperparametrai) ---
    models = get_rf_configurations()

    # 1) HOLD-OUT TRAIN/VALIDATION
    holdout_results, (X_tr, X_val, y_tr, y_val) = holdout_evaluate(
        models,
        X_train_val,
        y_train_val,
        val_size=0.2,
        random_state=42,
        stratify=True,
        average="weighted",
        verbose=True,
    )

    print("\n=== HOLD-OUT REZULTATAI (originalūs požymiai) ===")
    for name, metrics in holdout_results.items():
        print(f"\nModelis: {name}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 (weighted): {metrics['f1']:.4f}")

    # 2) K-FOLD CROSS-VALIDATION
    cv_results = crossval_evaluate(
        models,
        X_train_val,
        y_train_val,
        cv=5,
        random_state=42,
        average="weighted",
        verbose=True,
    )

    print("\n=== K-FOLD CROSS-VALIDATION REZULTATAI (originalūs požymiai) ===")
    for name, agg in cv_results.items():
        print(f"\nModelis: {name}")
        print(f"  accuracy_mean: {agg['accuracy_mean']:.4f} (std={agg['accuracy_std']:.4f})")
        print(f"  f1_mean      : {agg['f1_mean']:.4f} (std={agg['f1_std']:.4f})")

    # 3) PASIRENKAM GERIAUSIĄ KONFIGŪRACIJĄ PAGAL CV ACCURACY
    best_name = max(cv_results.keys(), key=lambda n: cv_results[n]["accuracy_mean"])
    print(f"\n>>> Pasirinktas geriausias modelis pagal CV: {best_name}")

    best_model = get_rf_configurations()[best_name]
    best_model.fit(X_train_val, y_train_val)

    y_pred_test = best_model.predict(X_test)
    print_test_metrics(
        "GALUTINIS TESTAVIMAS (originalūs 6 požymiai, klasifikavimo_test.csv)",
        y_test,
        y_pred_test,
    )


def experiment_tsne_features():
    """
    Eksperimentai su t-SNE 2D požymiais (classification_train_val_tsne.csv / classification_test_tsne.csv)
    Struktūra analogiška kaip originalių požymių atveju.
    """
    print("\n" + "#" * 60)
    print("RANDOM FOREST – t-SNE 2D POŽYMIAI")
    print("#" * 60)

    df_train_val = pd.read_csv("./data/classification_train_val_tsne.csv")
    df_test = pd.read_csv("./data/classification_test_tsne.csv")

    X_train_val = df_train_val.drop(columns=[TARGET_COL])
    y_train_val = df_train_val[TARGET_COL]

    X_test = df_test.drop(columns=[TARGET_COL])
    y_test = df_test[TARGET_COL]

    # t-SNE erdvėje požymių mažiau, bet naudosim tas pačias RF konfigūracijas
    models = get_rf_configurations()

    # 1) HOLD-OUT TRAIN/VALIDATION
    holdout_results, _ = holdout_evaluate(
        models,
        X_train_val,
        y_train_val,
        val_size=0.2,
        random_state=42,
        stratify=True,
        average="weighted",
        verbose=True,
    )

    print("\n=== HOLD-OUT REZULTATAI (t-SNE požymiai) ===")
    for name, metrics in holdout_results.items():
        print(f"\nModelis: {name}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 (weighted): {metrics['f1']:.4f}")

    # 2) K-FOLD CROSS-VALIDATION
    cv_results = crossval_evaluate(
        models,
        X_train_val,
        y_train_val,
        cv=5,
        random_state=42,
        average="weighted",
        verbose=True,
    )

    print("\n=== K-FOLD CROSS-VALIDATION REZULTATAI (t-SNE požymiai) ===")
    for name, agg in cv_results.items():
        print(f"\nModelis: {name}")
        print(f"  accuracy_mean: {agg['accuracy_mean']:.4f} (std={agg['accuracy_std']:.4f})")
        print(f"  f1_mean      : {agg['f1_mean']:.4f} (std={agg['f1_std']:.4f})")

    # 3) PASIRENKAM GERIAUSIĄ KONFIGŪRACIJĄ PAGAL CV ACCURACY
    best_name = max(cv_results.keys(), key=lambda n: cv_results[n]["accuracy_mean"])
    print(f"\n>>> Pasirinktas geriausias modelis pagal CV (t-SNE): {best_name}")

    best_model = get_rf_configurations()[best_name]
    best_model.fit(X_train_val, y_train_val)

    y_pred_test = best_model.predict(X_test)
    print_test_metrics(
        "GALUTINIS TESTAVIMAS (t-SNE 2D požymiai, classification_test_tsne.csv)",
        y_test,
        y_pred_test,
    )


def main():
    # Eksperimentai su originaliais 6 požymiais
    experiment_original_features()

    # Eksperimentai su t-SNE 2D požymiais
    experiment_tsne_features()


if __name__ == "__main__":
    main()
