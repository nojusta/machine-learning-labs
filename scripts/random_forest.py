import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

from evaluation import holdout_evaluate, crossval_evaluate

TARGET_COL = "NObeyesdad"
POS_LABEL = 5  # teigiama klasė ROC skaičiavimui
OUTPUT_DIR = "./outputs/random_forest"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------------------------------------------------------
# Pagalbinės metrikų ir vizualizacijų funkcijos
# -------------------------------------------------------------------

def print_test_metrics(title: str, y_true, y_pred) -> dict:
    """Atspausdina ir gražina pagrindines metrikas TEST rinkiniui."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)

    print(f"Accuracy       : {acc:.4f}")
    print(f"Precision (w)  : {precision:.4f}")
    print(f"Recall (w)     : {recall:.4f}")
    print(f"F1 (w)         : {f1:.4f}")
    print("\nConfusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(report)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "classification_report": report,
    }


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


def plot_confusion_matrix(y_true, y_pred, title: str, filename: str):
    plt.figure(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=[4, 5],
        cmap="Blues",
        colorbar=False,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches="tight")
    plt.show()


def plot_roc_curve(y_true, y_proba, title: str, filename: str) -> float:
    """Nubraižo ROC ir grąžina AUC."""
    y_bin = label_binarize(y_true, classes=[4, 5]).ravel()
    fpr, tpr, _ = roc_curve(y_bin, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches="tight")
    plt.show()

    return roc_auc


def plot_feature_importance(model: RandomForestClassifier, feature_names, title: str, filename: str):
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]

    sorted_names = [feature_names[i] for i in idx]
    sorted_vals = importances[idx]

    plt.figure(figsize=(6, 4))
    plt.bar(range(len(sorted_vals)), sorted_vals)
    plt.xticks(range(len(sorted_vals)), sorted_names, rotation=45, ha="right")
    plt.ylabel("Svarba")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches="tight")
    plt.show()


def analyze_errors(df_test: pd.DataFrame, y_true, y_pred, title: str, filename: str):
    """
    Paprasta klaidų analizė: kiek ir kokių sumaišymų tarp klasių,
    keli konkretūs pavyzdžiai.
    """
    print("\n" + "-" * 60)
    print(f"KLAIDŲ ANALIZĖ – {title}")
    print("-" * 60)

    df_err = df_test.copy()
    df_err["y_true"] = y_true.values
    df_err["y_pred"] = y_pred

    mis = df_err[df_err["y_true"] != df_err["y_pred"]]
    print(f"Bendras klaidingų klasifikacijų skaičius: {len(mis)} iš {len(df_test)}")

    print("\nKlaidų pasiskirstymas (tikros vs prognozuotos klasės):")
    print(pd.crosstab(mis["y_true"], mis["y_pred"]))

    # Išsaugom kelis pavyzdžius į CSV, kad būtų galima žiūrėti ataskaitai
    out_path = os.path.join(OUTPUT_DIR, filename)
    mis.head(20).to_csv(out_path, index=False)
    print(f"\nPirmi 20 klaidingai suklasifikuotų įrašų išsaugoti į: {out_path}")


def plot_tsne_decision_boundary(model: RandomForestClassifier,
                                df: pd.DataFrame,
                                title: str,
                                filename: str):
    """
    Sprendimo ribų vizualizacija t-SNE 2D erdvėje.
    Naudojam tsne_1 ir tsne_2 kaip X, o NObeyesdad kaip y.
    """
    X = df[["tsne_1", "tsne_2"]].values
    y = df[TARGET_COL].values

    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    # sprendimo ribos
    plt.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")

    # tikrieji testiniai taškai
    scatter = plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap="coolwarm",
        edgecolor="k",
        s=35,
        alpha=0.9,
    )
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title(title)
    handles, labels = scatter.legend_elements()
    plt.legend(handles, ["4 klasė", "5 klasė"], title="Tikra klasė", loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches="tight")
    plt.show()


# -------------------------------------------------------------------
# Eksperimentas su originaliais 6 požymiais
# -------------------------------------------------------------------

def experiment_original_features() -> dict:
    """
    Eksperimentai su pradiniu 6 požymių rinkiniu.
    Grąžina suvestines metrikas (hold-out, CV, test) šiam požymių rinkiniui.
    """
    print("\n" + "#" * 60)
    print("RANDOM FOREST – ORIGINALŪS 6 POŽYMIAI")
    print("#" * 60)

    # Duomenys
    df_train_val = pd.read_csv("./data/classification_train_val.csv")
    df_test = pd.read_csv("./data/classification_test.csv")

    X_train_val = df_train_val.drop(columns=[TARGET_COL])
    y_train_val = df_train_val[TARGET_COL]

    X_test = df_test.drop(columns=[TARGET_COL])
    y_test = df_test[TARGET_COL]

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

    # 3) Pasirenkam geriausią konfigūraciją pagal CV
    best_name = max(cv_results.keys(), key=lambda n: cv_results[n]["accuracy_mean"])
    print(f"\n>>> Pasirinktas geriausias modelis pagal CV: {best_name}")

    best_model = get_rf_configurations()[best_name]
    best_model.fit(X_train_val, y_train_val)

    # Prognozės ir tikimybės TEST rinkiniui
    y_pred_test = best_model.predict(X_test)
    classes = best_model.classes_
    pos_idx = np.where(classes == POS_LABEL)[0][0]
    y_proba_test = best_model.predict_proba(X_test)[:, pos_idx]

    # Tekstinės metrikos
    test_metrics = print_test_metrics(
        "GALUTINIS TESTAVIMAS (originalūs 6 požymiai, classification_test.csv)",
        y_test,
        y_pred_test,
    )

    # Vizualizacijos
    plot_confusion_matrix(
        y_test,
        y_pred_test,
        "Random Forest – Confusion Matrix (originalūs požymiai)",
        "original_confusion_matrix.png",
    )
    auc_original = plot_roc_curve(
        y_test,
        y_proba_test,
        "Random Forest – ROC kreivė (originalūs požymiai)",
        "original_roc_curve.png",
    )
    plot_feature_importance(
        best_model,
        X_train_val.columns,
        "Random Forest – požymių svarba (originalūs požymiai)",
        "original_feature_importance.png",
    )

    # Klaidų analizė (kur labiausiai klysta tarp 4 ir 5)
    analyze_errors(
        df_test,
        y_test,
        y_pred_test,
        "Originalūs požymiai (test rinkinys)",
        "original_misclassified_examples.csv",
    )

    # Suvestinė RF šiam požymių rinkiniui
    summary = {
        "feature_set": "original",
        "best_model": best_name,
        "holdout_accuracy": holdout_results[best_name]["accuracy"],
        "holdout_f1": holdout_results[best_name]["f1"],
        "cv_accuracy_mean": cv_results[best_name]["accuracy_mean"],
        "cv_f1_mean": cv_results[best_name]["f1_mean"],
        "test_accuracy": test_metrics["accuracy"],
        "test_f1": test_metrics["f1"],
        "test_auc": auc_original,
    }
    return summary


# -------------------------------------------------------------------
# Eksperimentas su t-SNE požymiais
# -------------------------------------------------------------------

def experiment_tsne_features() -> dict:
    """
    Eksperimentai su t-SNE 2D požymiais.
    Grąžina suvestines metrikas (hold-out, CV, test) šiam požymių rinkiniui.
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

    # 3) Pasirenkam geriausią konfigūraciją pagal CV
    best_name = max(cv_results.keys(), key=lambda n: cv_results[n]["accuracy_mean"])
    print(f"\n>>> Pasirinktas geriausias modelis pagal CV (t-SNE): {best_name}")

    best_model = get_rf_configurations()[best_name]
    best_model.fit(X_train_val, y_train_val)

    y_pred_test = best_model.predict(X_test)
    classes = best_model.classes_
    pos_idx = np.where(classes == POS_LABEL)[0][0]
    y_proba_test = best_model.predict_proba(X_test)[:, pos_idx]

    # Tekstinės metrikos
    test_metrics = print_test_metrics(
        "GALUTINIS TESTAVIMAS (t-SNE 2D požymiai, classification_test_tsne.csv)",
        y_test,
        y_pred_test,
    )

    # Vizualizacijos (klasifikavimo rezultatai)
    plot_confusion_matrix(
        y_test,
        y_pred_test,
        "Random Forest – Confusion Matrix (t-SNE 2D požymiai)",
        "tsne_confusion_matrix.png",
    )
    auc_tsne = plot_roc_curve(
        y_test,
        y_proba_test,
        "Random Forest – ROC kreivė (t-SNE 2D požymiai)",
        "tsne_roc_curve.png",
    )

    # t-SNE erdvės vizualizacija
    plt.figure(figsize=(5, 4))
    plt.scatter(
        df_test["tsne_1"],
        df_test["tsne_2"],
        c=y_test,
        cmap="coolwarm",
        alpha=0.75,
        edgecolor="k",
        s=40,
    )
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("t-SNE 2D erdvė (test rinkinys)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "tsne_2d_space_test.png"),
                dpi=300, bbox_inches="tight")
    plt.show()

    # Sprendimo ribos t-SNE 2D erdvėje (naudojam visus duomenis, kad būtų aiškesnė erdvė)
    df_all_tsne = pd.concat([df_train_val, df_test], ignore_index=True)
    plot_tsne_decision_boundary(
        best_model,
        df_all_tsne,
        "Random Forest – sprendimo ribos t-SNE 2D erdvėje",
        "tsne_decision_boundary.png",
    )

    # Klaidų analizė
    analyze_errors(
        df_test,
        y_test,
        y_pred_test,
        "t-SNE požymiai (test rinkinys)",
        "tsne_misclassified_examples.csv",
    )

    # Suvestinė RF šiam požymių rinkiniui
    summary = {
        "feature_set": "tsne_2d",
        "best_model": best_name,
        "holdout_accuracy": holdout_results[best_name]["accuracy"],
        "holdout_f1": holdout_results[best_name]["f1"],
        "cv_accuracy_mean": cv_results[best_name]["accuracy_mean"],
        "cv_f1_mean": cv_results[best_name]["f1_mean"],
        "test_accuracy": test_metrics["accuracy"],
        "test_f1": test_metrics["f1"],
        "test_auc": auc_tsne,
    }
    return summary


# -------------------------------------------------------------------
# Suvestinė lentelė (tik tekstu, be diagramų)
# -------------------------------------------------------------------

def summarize_rf_results(original_summary: dict, tsne_summary: dict):
    df_summary = pd.DataFrame([
        {
            "Požymių rinkinys": "Originalūs 6",
            "Geriausias modelis": original_summary["best_model"],
            "Hold-out acc": original_summary["holdout_accuracy"],
            "Hold-out F1": original_summary["holdout_f1"],
            "CV acc (mean)": original_summary["cv_accuracy_mean"],
            "CV F1 (mean)": original_summary["cv_f1_mean"],
            "Test acc": original_summary["test_accuracy"],
            "Test F1": original_summary["test_f1"],
            "Test AUC": original_summary["test_auc"],
        },
        {
            "Požymių rinkinys": "t-SNE 2D",
            "Geriausias modelis": tsne_summary["best_model"],
            "Hold-out acc": tsne_summary["holdout_accuracy"],
            "Hold-out F1": tsne_summary["holdout_f1"],
            "CV acc (mean)": tsne_summary["cv_accuracy_mean"],
            "CV F1 (mean)": tsne_summary["cv_f1_mean"],
            "Test acc": tsne_summary["test_accuracy"],
            "Test F1": tsne_summary["test_f1"],
            "Test AUC": tsne_summary["test_auc"],
        },
    ])

    print("\n" + "#" * 60)
    print("RANDOM FOREST REZULTATŲ SUVESTINĖ (originalūs vs t-SNE požymiai)")
    print("#" * 60)
    print(df_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Išsaugom ir į CSV, kad būtų galima įmesti į ataskaitos lentelę
    out_path = os.path.join(OUTPUT_DIR, "rf_results_summary.csv")
    df_summary.to_csv(out_path, index=False)
    print(f"\nSuvestinė išsaugota į: {out_path}")


def main():
    original_summary = experiment_original_features()
    tsne_summary = experiment_tsne_features()
    summarize_rf_results(original_summary, tsne_summary)


if __name__ == "__main__":
    main()