import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, auc, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ==========================================================
# 1. DUOMENŲ NUSKAITYMAS (ORIGINALŪS POŽYMIAI)
# ==========================================================

train_val = pd.read_csv("../data/classification_train_val.csv")
test = pd.read_csv("../data/classification_test.csv")

TARGET = "NObeyesdad"

X_train = train_val.drop(columns=[TARGET])
y_train = train_val[TARGET]

X_test = test.drop(columns=[TARGET])
y_test = test[TARGET]

# ==========================================================
# 2. HYPERPARAMETRŲ KONFIGŪRACIJOS
# ==========================================================

configs = {
    "dt_default": {
        "criterion": "gini",
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    },
    "dt_depth_3": {
        "criterion": "gini",
        "max_depth": 3,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    },
    "dt_depth_5": {
        "criterion": "gini",
        "max_depth": 5,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    },
    "dt_entropy_depth_5": {
        "criterion": "entropy",
        "max_depth": 5,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    },
    "dt_regularized": {
        "criterion": "gini",
        "max_depth": 6,
        "min_samples_split": 10,
        "min_samples_leaf": 5
    }
}


# ==========================================================
# 3. FUNKCIJA MODELIUI VERTINTI
# ==========================================================

def evaluate_model(name, params, X_train, y_train, X_test, y_test):
    model = DecisionTreeClassifier(**params, random_state=42)

    # --- Hold-out ---
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=5)
    rec = recall_score(y_test, y_pred, pos_label=5)
    f1 = f1_score(y_test, y_pred, pos_label=5)

    fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=5)
    auc_score = auc(fpr, tpr)

    # --- Cross-validation (5-fold) ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_acc_list = []
    cv_prec_list = []
    cv_rec_list = []
    cv_f1_list = []
    cv_auc_list = []

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model_cv = DecisionTreeClassifier(**params, random_state=42)
        model_cv.fit(X_tr, y_tr)

        y_val_pred = model_cv.predict(X_val)
        y_val_proba = model_cv.predict_proba(X_val)[:, 1]

        cv_acc_list.append(accuracy_score(y_val, y_val_pred))
        cv_prec_list.append(precision_score(y_val, y_val_pred, pos_label=5))
        cv_rec_list.append(recall_score(y_val, y_val_pred, pos_label=5))
        cv_f1_list.append(f1_score(y_val, y_val_pred, pos_label=5))

        fpr_cv, tpr_cv, _ = roc_curve(y_val, y_val_proba, pos_label=5)
        cv_auc_list.append(auc(fpr_cv, tpr_cv))

    return {
        "name": name,
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "auc": auc_score,
        "cv_accuracy": np.mean(cv_acc_list),
        "cv_precision": np.mean(cv_prec_list),
        "cv_recall": np.mean(cv_rec_list),
        "cv_f1": np.mean(cv_f1_list),
        "cv_auc": np.mean(cv_auc_list),
        "model": model,
        "fpr": fpr,
        "tpr": tpr,
        "y_pred": y_pred
    }



# ==========================================================
# 4. PALEIDŽIAMI VISI MODELIAI (ORIGINALŪS POŽYMIAI)
# ==========================================================

results = []
for name, params in configs.items():
    res = evaluate_model(name, params, X_train, y_train, X_test, y_test)
    print(f"\n=== Modelis: {name} ===")
    print(res)
    results.append(res)

df_results = pd.DataFrame(results)
print("\n=== Rezultatų lentelė (originalūs požymiai) ===")
print(df_results)

# ==========================================================
# 5. PASIRENKAM GERIAUSIĄ MODELĮ (ORIGINALŪS POŽYMIAI)
# ==========================================================

best = max(results, key=lambda x: x["auc"])
best_model = best["model"]
best_model.fit(X_train, y_train)

print("\n=== GERIAUSIAS MODELIS PAGAL AUC (originalūs požymiai) ===")
print(best["name"], best["auc"])

# ==========================================================
# 6. VIZUALIZACIJOS (ORIGINALŪS POŽYMIAI)
# ==========================================================

# Confusion Matrix
cm = confusion_matrix(y_test, best["y_pred"])
print("\n=== Confusion Matrix with labels ===")
print("          Pred 4   Pred 5")
print(f"True 4     {cm[0][0]:5d}   {cm[0][1]:5d}")
print(f"True 5     {cm[1][0]:5d}   {cm[1][1]:5d}")

plt.figure(figsize=(6, 6))
plt.imshow(cm, cmap="Blues")
plt.xticks([0, 1], [4, 5])
plt.yticks([0, 1], [4, 5])
plt.title(f"Decision Tree – Confusion Matrix (6 požymiai)")

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)

plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()

# ROC Curve
plt.figure(figsize=(6, 5))
plt.plot(best["fpr"], best["tpr"], label=f"AUC = {best['auc']:.3f}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.title(f"ROC Curve – Decision Tree (6 požymiai)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# ==========================================================
# === === ===   t-SNE  ANALIZĖ   === === ===
# ==========================================================

print("\n\n====================")
print("   T-SNE ANALIZĖ")
print("====================\n")

# 1. Load t-SNE data
train_val_tsne = pd.read_csv("../data/classification_train_val_tsne.csv")
test_tsne = pd.read_csv("../data/classification_test_tsne.csv")

X_train_tsne = train_val_tsne.drop(columns=[TARGET])
y_train_tsne = train_val_tsne[TARGET]

X_test_tsne = test_tsne.drop(columns=[TARGET])
y_test_tsne = test_tsne[TARGET]

# 2. Evaluate models on t-SNE
results_tsne = []
for name, params in configs.items():
    res = evaluate_model(name, params, X_train_tsne, y_train_tsne, X_test_tsne, y_test_tsne)
    print(f"\n=== t-SNE modelis: {name} ===")
    print(res)
    results_tsne.append(res)

df_results_tsne = pd.DataFrame(results_tsne)
print("\n=== Rezultatų lentelė (t-SNE) ===")
print(df_results_tsne)

# 3. Pick best model
best_tsne = max(results_tsne, key=lambda x: x["auc"])
print("\n=== GERIAUSIAS MODELIS PAGAL AUC (t-SNE) ===")
print(best_tsne["name"], best_tsne["auc"])

# 4. Confusion Matrix (t-SNE)
cm = confusion_matrix(y_test_tsne, best_tsne["y_pred"])
print("\n=== Confusion Matrix with labels ===")
print("          Pred 4   Pred 5")
print(f"True 4     {cm[0][0]:5d}   {cm[0][1]:5d}")
print(f"True 5     {cm[1][0]:5d}   {cm[1][1]:5d}")

plt.figure(figsize=(6, 6))
plt.imshow(cm, cmap="Blues")
plt.xticks([0, 1], [4, 5])
plt.yticks([0, 1], [4, 5])
plt.title(f"Decision Tree – Confusion Matrix (t-SNE)")

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)

plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()

# ==========================================================
# 4B. VERTINIMO STRATEGIJŲ PALYGINIMO LENTELĖ
# ==========================================================

summary_rows = []

for res in results:
    model = res["model"]

    # Train accuracy
    y_pred_train = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_pred_train)

    summary_rows.append({
        "Modelis": res["name"],
        "Hold-out Accuracy": res["acc"],
        "Hold-out Precision": res["prec"],
        "Hold-out Recall": res["rec"],
        "Hold-out F1": res["f1"],
        "Hold-out AUC": res["auc"],

        "CV Accuracy": res["cv_accuracy"],
        "CV Precision": res["cv_precision"],
        "CV Recall": res["cv_recall"],
        "CV F1": res["cv_f1"],
        "CV AUC": res["cv_auc"],
    })

df_summary = pd.DataFrame(summary_rows)
print("\n=== Apibendrinta vertinimo strategijų lentelė ===")
print(df_summary.to_string(index=False))

# 5. ROC Curve (t-SNE)
plt.figure(figsize=(6, 5))
plt.plot(best_tsne["fpr"], best_tsne["tpr"], label=f"AUC = {best_tsne['auc']:.3f}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.title(f"ROC Curve – Decision Tree (t-SNE)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ==========================================================
# === 7. SPRENDIMO RIBŲ VIZUALIZACIJA (BOUNDARY PLOTS) ===
# ==========================================================

import os
import matplotlib.patches as mpatches

output_dir = "../outputs/decision_tree"
os.makedirs(output_dir, exist_ok=True)

# Bendra legenda
handles = [
    mpatches.Patch(color="#4575b4", label="Klasė 4"),
    mpatches.Patch(color="#d73027", label="Klasė 5")
]

# ----------------------------
# 7A — ORIGINAL TRAIN BOUNDARY
# ----------------------------

feat_x = "FCVC"
feat_y = "FAF"

X_train_2d = X_train[[feat_x, feat_y]]

model_2d_train = DecisionTreeClassifier(
    criterion=best_model.criterion,
    max_depth=best_model.max_depth,
    min_samples_split=best_model.min_samples_split,
    min_samples_leaf=best_model.min_samples_leaf,
    random_state=42
)
model_2d_train.fit(X_train_2d, y_train)

x_min, x_max = X_train_2d[feat_x].min() - 0.05, X_train_2d[feat_x].max() + 0.05
y_min, y_max = X_train_2d[feat_y].min() - 0.05, X_train_2d[feat_y].max() + 0.05

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 400),
    np.linspace(y_min, y_max, 400)
)

Z = model_2d_train.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.35, cmap="coolwarm")
plt.scatter(X_train_2d[feat_x], X_train_2d[feat_y], c=y_train, cmap="coolwarm", edgecolor="black")
plt.xlabel(feat_x)
plt.ylabel(feat_y)
plt.title("Boundary – TRAIN (FCVC × FAF)")
plt.legend(handles=handles, title="Klasė")

plt.savefig(f"{output_dir}/boundary_train_fcvc_faf.png", dpi=300, bbox_inches="tight")
plt.show()


# ----------------------------
# 7B — ORIGINAL TEST BOUNDARY
# ----------------------------

X_test_2d = X_test[[feat_x, feat_y]]

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.35, cmap="coolwarm")  # Naudojame TRAIN ribas
plt.scatter(X_test_2d[feat_x], X_test_2d[feat_y], c=y_test, cmap="coolwarm", edgecolor="black")
plt.xlabel(feat_x)
plt.ylabel(feat_y)
plt.title("Boundary – TEST (FCVC × FAF)")
plt.legend(handles=handles, title="Klasė")

plt.savefig(f"{output_dir}/boundary_test_fcvc_faf.png", dpi=300, bbox_inches="tight")
plt.show()


# ==========================================================
# === t-SNE BOUNDARY – TRAIN
# ==========================================================

feat_x = "tsne_1"
feat_y = "tsne_2"

X_train_tsne_2d = X_train_tsne[[feat_x, feat_y]]

model_tsne_2d_train = DecisionTreeClassifier(
    criterion=best_tsne["model"].criterion,
    max_depth=best_tsne["model"].max_depth,
    min_samples_split=best_tsne["model"].min_samples_split,
    min_samples_leaf=best_tsne["model"].min_samples_leaf,
    random_state=42
)
model_tsne_2d_train.fit(X_train_tsne_2d, y_train_tsne)

x_min, x_max = X_train_tsne_2d[feat_x].min() - 1, X_train_tsne_2d[feat_x].max() + 1
y_min, y_max = X_train_tsne_2d[feat_y].min() - 1, X_train_tsne_2d[feat_y].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 600),
    np.linspace(y_min, y_max, 600)
)

Z = model_tsne_2d_train.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.35, cmap="coolwarm")
plt.scatter(X_train_tsne_2d[feat_x], X_train_tsne_2d[feat_y], c=y_train_tsne, cmap="coolwarm", edgecolor="black")
plt.xlabel(feat_x)
plt.ylabel(feat_y)
plt.title("Boundary – t-SNE TRAIN")
plt.legend(handles=handles, title="Klasė")

plt.savefig(f"{output_dir}/boundary_tsne_train.png", dpi=300, bbox_inches="tight")
plt.show()


# ==========================================================
# === t-SNE BOUNDARY – TEST
# ==========================================================

X_test_tsne_2d = X_test_tsne[[feat_x, feat_y]]

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.35, cmap="coolwarm")
plt.scatter(X_test_tsne_2d[feat_x], X_test_tsne_2d[feat_y], c=y_test_tsne, cmap="coolwarm", edgecolor="black")
plt.xlabel(feat_x)
plt.ylabel(feat_y)
plt.title("Boundary – t-SNE TEST")
plt.legend(handles=handles, title="Klasė")

plt.savefig(f"{output_dir}/boundary_tsne_test.png", dpi=300, bbox_inches="tight")
plt.show()
