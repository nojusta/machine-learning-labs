import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
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
from sklearn.base import clone

from evaluation import holdout_evaluate, crossval_evaluate

# ----- BENDRI NUSTATYMAI -----
TARGET_COL = "NObeyesdad"
POS_LABEL = 5  # teigiama klasė ROC/AUC skaičiavimui

OUTPUT_DIR = "./outputs/random_forest"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# PAGALBINĖS FUNKCIJOS METRIKOMS IR VIZUALIZACIJOMS
# ============================================================

def print_test_metrics(title: str, y_true, y_pred) -> dict:
    """
    Pagalbinė funkcija galutinio TEST rinkinio metrikoms atspausdinti.
    Grąžina ir metrikų žodyną, kad galėtume naudoti lentelėms.
    """
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

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
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

#================================================================================
    #PAGALBINĖ FUNKCIJA OOB METRIKOMS APSKAIČIUOTI (palyginimui su CV/holdout)
#================================================================================

def compute_oob_metrics(model: RandomForestClassifier, y_true, pos_label: int = POS_LABEL) -> dict:
    """
    Apskaičiuoja OOB metrikas iš jau apmokyto RandomForest su oob_score=True.
    Grąžina accuracy, precision, recall, F1 ir AUC (pagal pos_label).
    """
    if not getattr(model, "oob_score", False):
        raise ValueError("Modelis turi būti apmokytas su oob_score=True")

    # OOB tikimybės (eilutėms iš train duomenų, bet skaičiuota tik iš medžių,
    # kurie NEMATĖ konkretaus stebėjimo mokymo metu)
    proba = model.oob_decision_function_      # shape: (n_samples, n_classes)
    classes = model.classes_
    oob_pred_idx = np.argmax(proba, axis=1)
    y_pred = classes[oob_pred_idx]

    # klasės tikimybės AUC skaičiavimui
    pos_idx = np.where(classes == pos_label)[0][0]
    y_proba = proba[:, pos_idx]

    # klasikinės metrikos
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # AUC
    y_bin = label_binarize(y_true, classes=[4, 5]).ravel()
    fpr, tpr, _ = roc_curve(y_bin, y_proba)
    roc_auc = auc(fpr, tpr)

    print("\n--- OOB validacija (train_val aibė) ---")
    print(f"OOB Accuracy   : {acc:.4f}")
    print(f"OOB Precision  : {precision:.4f}")
    print(f"OOB Recall     : {recall:.4f}")
    print(f"OOB F1 (w)     : {f1:.4f}")
    print(f"OOB AUC        : {roc_auc:.4f}")

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": roc_auc,
    }

# ----- VIZUALIZACIJOS -----

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
    """
    Nubraižo ROC kreivę ir grąžina AUC (kad paskui galėtume įdėti į lentelę).
    """
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


def plot_decision_boundary_2d(model, X_2d, y, title: str, filename: str):
    """
    Sprendimo ribų brėžinys 2D erdvėje.
    X_2d – DataFrame su 2 požymiais.
    """
    assert X_2d.shape[1] == 2, "Decision boundary funkcija skirta tik 2D požymiams."

    x_min, x_max = X_2d.iloc[:, 0].min() - 0.1, X_2d.iloc[:, 0].max() + 0.1
    y_min, y_max = X_2d.iloc[:, 1].min() - 0.1, X_2d.iloc[:, 1].max() + 0.1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    scatter = plt.scatter(
        X_2d.iloc[:, 0],
        X_2d.iloc[:, 1],
        c=y,
        cmap="coolwarm",
        edgecolor="k",
        s=40,
        alpha=0.8,
    )
    plt.xlabel(X_2d.columns[0])
    plt.ylabel(X_2d.columns[1])
    plt.title(title)
    plt.legend(*scatter.legend_elements(), title="Klasė", loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches="tight")
    plt.show()


def export_misclassified(df_features: pd.DataFrame, y_true, y_pred, filename: str):
    """
    Išsaugo klaidingai suklasifikuotus taškus (ypač kur 4 ir 5 sumaišomos).
    """
    df_err = df_features.copy()
    df_err["true_label"] = y_true
    df_err["pred_label"] = y_pred
    df_err = df_err[df_err["true_label"] != df_err["pred_label"]]

    out_path = os.path.join(OUTPUT_DIR, filename)
    df_err.to_csv(out_path, index=False)

    print(f"\nKlaidingų klasifikacijų: {len(df_err)} / {len(df_features)}")
    print(f"Išsaugota į: {out_path}")

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

# ============================================================
# EKSPERIMENTAS SU ORIGINALIAIS 6 POŽYMIAIS
# ============================================================

def experiment_original_features():
    """
    Eksperimentai su pradiniu 6 požymių rinkiniu
    (classification_train_val.csv / classification_test.csv):

      1) Hold-out train/validation
      2) k-fold kryžminė validacija
      3) Galutinis testavimas ant classification_test.csv
      4) ROC + AUC, sumaišymo matrica, požymių svarba
      5) 2D sprendimo ribos pagal 2 svarbiausius požymius
      6) Klaidingų klasifikacijų eksportas
      7) RF rezultatų santraukos lentelė (holdout / CV / test)
    """
    print("\n" + "#" * 60)
    print("RANDOM FOREST – ORIGINALŪS 6 POŽYMIAI")
    print("#" * 60)

    # --- Duomenys ---
    df_train_val = pd.read_csv("./data/classification_train_val.csv")
    df_test = pd.read_csv("./data/classification_test.csv")

    X_train_val = df_train_val.drop(columns=[TARGET_COL])
    y_train_val = df_train_val[TARGET_COL]

    X_test = df_test.drop(columns=[TARGET_COL])
    y_test = df_test[TARGET_COL]

    models = get_rf_configurations()

    # 1) HOLD-OUT
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

    # 2) K-FOLD CROSS-VAL
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

    # 3) GERIAUSIAS MODELIS PAGAL CV ACCURACY
    best_name = max(cv_results.keys(), key=lambda n: cv_results[n]["accuracy_mean"])
    print(f"\n>>> Pasirinktas geriausias modelis pagal CV: {best_name}")

    best_model = get_rf_configurations()[best_name]
    best_model.fit(X_train_val, y_train_val)

    # Prognozės ir tikimybės TEST rinkinyje
    y_pred_test = best_model.predict(X_test)
    classes = best_model.classes_
    pos_idx = np.where(classes == POS_LABEL)[0][0]
    y_proba_test = best_model.predict_proba(X_test)[:, pos_idx]

    # 4) Tekstinės metrikos + ROC/AUC + sumaišymo matrica + feature importance
    test_metrics = print_test_metrics(
        "GALUTINIS TESTAVIMAS (originalūs 6 požymiai, classification_test.csv)",
        y_test,
        y_pred_test,
    )

    auc_test = plot_roc_curve(
        y_test,
        y_proba_test,
        "Random Forest – ROC kreivė (originalūs požymiai)",
        "original_roc_curve.png",
    )
    plot_confusion_matrix(
        y_test,
        y_pred_test,
        "Random Forest – Confusion Matrix (originalūs požymiai)",
        "original_confusion_matrix.png",
    )
    plot_feature_importance(
        best_model,
        X_train_val.columns,
        "Random Forest – požymių svarba (originalūs požymiai)",
        "original_feature_importance.png",
    )

    # 5) Sprendimo ribos 2D pagal 2 svarbiausius požymius
    importances = best_model.feature_importances_
    top2_idx = np.argsort(importances)[::-1][:2]
    top2_features = X_train_val.columns[top2_idx]
    print(f"\n2 svarbiausi požymiai sprendimo ribų vizualizacijai: {list(top2_features)}")

    # naujas RF tik su 2 požymiais, kad sprendimo riba būtų švariai 2D
    rf_2d = clone(best_model)
    rf_2d.fit(X_train_val[top2_features], y_train_val)

    X_test_2d = X_test[top2_features]
    plot_decision_boundary_2d(
        rf_2d,
        X_test_2d,
        y_test,
        f"RF sprendimo ribos 2D (originalūs požymiai: {top2_features[0]}, {top2_features[1]})",
        "original_decision_boundary.png",
    )

    X_train_val_2d = X_train_val[top2_features]
    plot_decision_boundary_2d(
        rf_2d,
        X_train_val_2d,
        y_train_val,
        f"RF sprendimo ribos 2D (train+val, požymiai: {top2_features[0]}, {top2_features[1]})",
       "original_decision_boundary_trainval.png",
    )

    # 6) Klaidingų klasifikacijų eksportas (pilni 6 požymiai)
    export_misclassified(
        X_test,
        y_test,
        y_pred_test,
        "misclassified_original_features.csv",
    )

    error_report = analyze_errors(
        X_test,
        y_test,
        y_pred_test.tolist(),
        y_proba_test.tolist(),
        POS_LABEL,
        Path(OUTPUT_DIR) / "original_misclassified_examples.csv",
    )
    print("\n--- ERROR ANALYSIS (originalūs požymiai) ---")
    print(f"Klaidingų prognozių: {error_report['total_misclassified']}")
    print(f"Prie sprendimo ribos (p∈[0.4,0.6]): {error_report['fraction_near_boundary_prob_0.4_0.6']:.2f}")
    print(f"False positives: {error_report['false_positive']['count']}")
    print(f"False negatives: {error_report['false_negative']['count']}")
    print(f"CSV: {error_report['csv_path']}")

    # --- 6.5) OOB validacija su tuo pačiu geriausiu RF ---
    # Kuriame kopiją su oob_score=True (bootstrap paliekam True – default)
    best_model_oob = clone(best_model)
    best_model_oob.set_params(oob_score=True, bootstrap=True)

    # OOB logika skaičiuojama ant VISOS train_val aibės (be skaidymo)
    best_model_oob.fit(X_train_val, y_train_val)
    oob_metrics = compute_oob_metrics(best_model_oob, y_train_val, pos_label=POS_LABEL)

    # 7) RF rezultatų SANTRAUKOS lentelė (holdout / CV / test)
    summary_rows = []

    # OOB – vidinė RF validacija
    summary_rows.append({
        "setup": "oob_val",
        "accuracy": oob_metrics["accuracy"],
        "precision": oob_metrics["precision"],
        "recall": oob_metrics["recall"],
        "f1": oob_metrics["f1"],
        "auc": oob_metrics["auc"],
    })

    # holdout – imame geriausio modelio metrikas
    best_holdout = holdout_results[best_name]
    summary_rows.append({
        "setup": "holdout_val",
        "accuracy": best_holdout["accuracy"],
        "precision": best_holdout["precision"],
        "recall": best_holdout["recall"],
        "f1": best_holdout["f1"],
        "auc": np.nan,  # AUC formaliai neskaičiuotas holdout'e
    })

    # cv – vidurkiai
    best_cv = cv_results[best_name]
    summary_rows.append({
        "setup": "cv_val_mean",
        "accuracy": best_cv["accuracy_mean"],
        "precision": best_cv["precision_mean"],
        "recall": best_cv["recall_mean"],
        "f1": best_cv["f1_mean"],
        "auc": np.nan,
    })

    # test – pilnos metrikos + AUC
    summary_rows.append({
        "setup": "test",
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "auc": auc_test,
    })

    df_summary = pd.DataFrame(summary_rows)
    print("\n=== RF SANTRAUKA (originalūs požymiai) ===")
    print(df_summary)

    df_summary.to_csv(
        os.path.join(OUTPUT_DIR, "rf_summary_original_features.csv"),
        index=False,
    )


# ============================================================
# EKSPERIMENTAS SU t-SNE 2D POŽYMIAIS
# ============================================================

def experiment_tsne_features():
    """
    Eksperimentai su t-SNE 2D požymiais
    (classification_train_val_tsne.csv / classification_test_tsne.csv):

      1) Hold-out train/validation
      2) k-fold kryžminė validacija
      3) Galutinis testavimas + ROC/AUC + sumaišymo matrica
      4) Sprendimo ribos t-SNE 2D erdvėje
      5) Klaidingų klasifikacijų eksportas
      6) RF rezultatų santraukos lentelė
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

    # 1) HOLD-OUT
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

    # 2) K-FOLD CROSS-VAL
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

    # 3) GERIAUSIAS MODELIS PAGAL CV
    best_name = max(cv_results.keys(), key=lambda n: cv_results[n]["accuracy_mean"])
    print(f"\n>>> Pasirinktas geriausias modelis pagal CV (t-SNE): {best_name}")

    best_model = get_rf_configurations()[best_name]
    best_model.fit(X_train_val, y_train_val)

    y_pred_test = best_model.predict(X_test)
    classes = best_model.classes_
    pos_idx = np.where(classes == POS_LABEL)[0][0]
    y_proba_test = best_model.predict_proba(X_test)[:, pos_idx]

    # Tekstinės metrikos + ROC/AUC + CM
    test_metrics = print_test_metrics(
        "GALUTINIS TESTAVIMAS (t-SNE 2D požymiai, classification_test_tsne.csv)",
        y_test,
        y_pred_test,
    )

    auc_test = plot_roc_curve(
        y_test,
        y_proba_test,
        "Random Forest – ROC kreivė (t-SNE 2D požymiai)",
        "tsne_roc_curve.png",
    )
    plot_confusion_matrix(
        y_test,
        y_pred_test,
        "Random Forest – Confusion Matrix (t-SNE 2D požymiai)",
        "tsne_confusion_matrix.png",
    )

    # 4) Sprendimo ribos t-SNE 2D erdvėje
    plot_decision_boundary_2d(
        best_model,
        X_test[["tsne_1", "tsne_2"]],
        y_test,
        "RF sprendimo ribos t-SNE 2D erdvėje (test rinkinys)",
        "tsne_decision_boundary.png",
    )

    plot_decision_boundary_2d(
        best_model,
        X_train_val[["tsne_1", "tsne_2"]],
        y_train_val,
        "RF sprendimo ribos t-SNE 2D erdvėje (train+val aibė)",
        "tsne_decision_boundary_trainval.png",
    )

    # 5) Klaidingų klasifikacijų eksportas t-SNE erdvėje
    export_misclassified(
        X_test, 
        y_test,
        y_pred_test,
        "misclassified_tsne_features.csv",
    )

    error_report = analyze_errors(
        X_test,
        y_test,
        y_pred_test.tolist(),
        y_proba_test.tolist(),
        POS_LABEL,
        Path(OUTPUT_DIR) / "tsne_misclassified_examples.csv",
    )
    print("\n--- ERROR ANALYSIS (t-SNE požymiai) ---")
    print(f"Klaidingų prognozių: {error_report['total_misclassified']}")
    print(f"Prie sprendimo ribos (p∈[0.4,0.6]): {error_report['fraction_near_boundary_prob_0.4_0.6']:.2f}")
    print(f"False positives: {error_report['false_positive']['count']}")
    print(f"False negatives: {error_report['false_negative']['count']}")
    print(f"CSV: {error_report['csv_path']}")

    #OOB

    best_model_oob = clone(best_model)
    best_model_oob.set_params(oob_score=True, bootstrap=True)
    best_model_oob.fit(X_train_val, y_train_val)
    oob_metrics = compute_oob_metrics(best_model_oob, y_train_val, pos_label=POS_LABEL)

    # 6) RF SANTRAUKA t-SNE
    summary_rows = []

    summary_rows.append({
        "setup": "oob_val_tsne",
        "accuracy": oob_metrics["accuracy"],
        "precision": oob_metrics["precision"],
        "recall": oob_metrics["recall"],
        "f1": oob_metrics["f1"],
        "auc": oob_metrics["auc"],
    })

    best_holdout = holdout_results[best_name]
    summary_rows.append({
        "setup": "holdout_val_tsne",
        "accuracy": best_holdout["accuracy"],
        "precision": best_holdout["precision"],
        "recall": best_holdout["recall"],
        "f1": best_holdout["f1"],
        "auc": np.nan,
    })

    best_cv = cv_results[best_name]
    summary_rows.append({
        "setup": "cv_val_mean_tsne",
        "accuracy": best_cv["accuracy_mean"],
        "precision": best_cv["precision_mean"],
        "recall": best_cv["recall_mean"],
        "f1": best_cv["f1_mean"],
        "auc": np.nan,
    })

    summary_rows.append({
        "setup": "test_tsne",
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "auc": auc_test,
    })

    df_summary = pd.DataFrame(summary_rows)
    print("\n=== RF SANTRAUKA (t-SNE požymiai) ===")
    print(df_summary)

    df_summary.to_csv(
        os.path.join(OUTPUT_DIR, "rf_summary_tsne_features.csv"),
        index=False,
    )

    # papildomai – tik vizualus t-SNE pasiskirstymas (tikrų klasių)
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
    plt.savefig(
        os.path.join(OUTPUT_DIR, "tsne_2d_space_test.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    # Eksperimentai su originaliais 6 požymiais
    experiment_original_features()

    # Eksperimentai su t-SNE 2D požymiais
    experiment_tsne_features()


if __name__ == "__main__":
    main()
