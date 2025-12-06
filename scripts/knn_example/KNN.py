import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

# ---------- KONSTANTOS IR NUSTATYMAI ----------
DUOMENU_DIREKTORIJA = '../duomenys'
GRAFIKU_DIREKTORIJA = '../grafikai'
KNN_DIREKTORIJA = 'KNN'
JSON_DIREKTORIJA = '../JSON'
JSON_FAILAS = os.path.join(JSON_DIREKTORIJA, 'geriausias_rinkinys.json')

# Sukuriame reikiamas direktorijas
os.makedirs(os.path.join(GRAFIKU_DIREKTORIJA, KNN_DIREKTORIJA), exist_ok=True)

# ---------- 1. DUOMENU IKELIMAS ----------
print("=" * 100)
print(" 1. DUOMENU IKELIMAS IR PARUOSIMAS ".center(100, "="))

try:
    df_mokymas = pd.read_csv(os.path.join(DUOMENU_DIREKTORIJA, 'mokymo_aibe.csv'), sep=';')
    df_validavimas = pd.read_csv(os.path.join(DUOMENU_DIREKTORIJA, 'validavimo_aibe.csv'), sep=';')
    df_testavimas = pd.read_csv(os.path.join(DUOMENU_DIREKTORIJA, 'testavimo_aibe.csv'), sep=';')
    print(f"[OK] Duomenys ikelti. Mokymo imtis: {len(df_mokymas)} eiluciu.")
except FileNotFoundError:
    print(f"[KLAIDA] Nerasti duomenys aplanke '{DUOMENU_DIREKTORIJA}'.")
    exit()

# Isskiriame Target kintamaji (klases)
y_mokymas = df_mokymas['label'].values
y_validavimas = df_validavimas['label'].values
y_testavimas = df_testavimas['label'].values

# ---------- 2. EKSPERIMENTU APIBREZIMAS ----------

pozymiai_full = [col for col in df_mokymas.columns if col != 'label']

try:
    with open(JSON_FAILAS, 'r', encoding='utf-8') as f:
        config = json.load(f)
        pozymiai_subset = config.get("GERIAUSIAS_MODELIS_6_POZYMIAI", [])
        print(f"[OK] Ikelti geriausi pozymiai is JSON ({len(pozymiai_subset)} vnt.)")
except FileNotFoundError:
    print("[INFO] JSON nerastas. Naudojami numatytieji QRS pozymiai.")
    pozymiai_subset = ["Q_val", "R_val", "S_val", "Q_pos", "R_pos", "S_pos"]

experiments = {
    "Visi požymiai": pozymiai_full,
    "Optimalūs požymiai": pozymiai_subset
}

# Čia nurodyti fiksuoti k parametrai specifinėms aibėms
FIXED_K_PARAMS = {
    "Optimalūs požymiai": 3
}

# Sąrašai duomenų kaupimui
roc_data_storage = []
cm_data_storage = [] 
summary_results = []
visu_eksperimentu_duomenys = []

# ---------- 3. PAGRINDINIS CIKLAS ----------

for exp_name, features in experiments.items():
    print("\n" + "#" * 100)
    print(f" VYKDOMAS EKSPERIMENTAS: {exp_name} ".center(100, "#"))

    X_mok = df_mokymas[features].values
    X_val = df_validavimas[features].values
    X_test = df_testavimas[features].values

    print(f"--- Ieskomas geriausias k (Validavimo aibe) ---")
    
    # Kintamieji geriausio paieškai (automatiniai)
    auto_best_k = 1
    best_val_f1 = -1
    tuning_data_table = []

    # Ciklas vykdomas visada, kad užpildytume duomenis grafikams (Section 5)
    for k in range(1, 22, 2):
        knn_temp = KNeighborsClassifier(n_neighbors=k, metric='euclidean', weights='uniform')
        knn_temp.fit(X_mok, y_mokymas)
        y_val_pred = knn_temp.predict(X_val)

        acc_val = accuracy_score(y_validavimas, y_val_pred)
        prec_val = precision_score(y_validavimas, y_val_pred, average='weighted', zero_division=0)
        rec_val = recall_score(y_validavimas, y_val_pred, average='weighted', zero_division=0)
        f1_val = f1_score(y_validavimas, y_val_pred, average='weighted', zero_division=0)

        tuning_data_table.append([k, acc_val, prec_val, rec_val, f1_val])

        # Kaupiame duomenis grafikams
        visu_eksperimentu_duomenys.append({
            'Dataset': exp_name,
            'k': k,
            'Accuracy': acc_val,
            'Precision': prec_val,
            'Recall': rec_val,
            'F1 Score': f1_val
        })

        if f1_val > best_val_f1:
            best_val_f1 = f1_val
            auto_best_k = k

    print(f"\nParametru paieskos rezultatai ({exp_name}):")
    headers = ["k", "Accuracy", "Precision", "Recall", "F1 Score"]
    print(tabulate(tuning_data_table, headers=headers, tablefmt="psql", floatfmt=".4f"))

    # --- SPRENDIMAS DĖL GERIAUSIO K ---
    if exp_name in FIXED_K_PARAMS:
        best_k = FIXED_K_PARAMS[exp_name]
        print(f"\n[INFO] Rastas automatinis k={auto_best_k}, BET naudojamas FIKSUOTAS k={best_k} (pagal nustatymus).")
    else:
        best_k = auto_best_k
        print(f"\n[BEST] Automatiškai pasirinktas k: {best_k} (Maksimalus F1={best_val_f1:.4f})")

    # -----------------------------------------------------------
    # GALUTINIS TESTAVIMAS SU TESTAVIMO AIBE
    # -----------------------------------------------------------
    final_knn = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean', weights='uniform')
    final_knn.fit(X_mok, y_mokymas)
    y_test_pred = final_knn.predict(X_test)

    # Metrikos
    acc = accuracy_score(y_testavimas, y_test_pred)
    prec = precision_score(y_testavimas, y_test_pred, average='weighted', zero_division=0)
    rec = recall_score(y_testavimas, y_test_pred, average='weighted', zero_division=0)
    f1_final = f1_score(y_testavimas, y_test_pred, average='weighted', zero_division=0)

    summary_results.append({
        'Dataset': exp_name,
        'k': best_k,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1_final
    })

    # --- SVARBU: Išvedame detalią ataskaitą (Classification Report) ---
    print(f"\n>>> DETALI KLASIFIKAVIMO ATASKAITA: {exp_name} (k={best_k}) <<<")
    print(classification_report(y_testavimas, y_test_pred, target_names=["Normalus (0)", "Aritmija (2)"], digits=4))

    # --- SVARBU: Sumaišymo matricos skaičiai tekstui ---
    cm = confusion_matrix(y_testavimas, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f">>> KLAIDŲ ANALIZĖ: TN={tn} (Tikrai sveiki), FP={fp} (Klaidingi aliarmai), FN={fn} (Praleista liga), TP={tp} (Rasta liga)")
    print("-" * 60)

    cm_data_storage.append({
        'cm': cm,
        'title': f'{exp_name}\nk={best_k}'
    })

    # ROC Data
    if hasattr(final_knn, "predict_proba"):
        y_proba = final_knn.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_testavimas, y_proba, pos_label=2)
        roc_auc = auc(fpr, tpr)
        roc_data_storage.append({'name': exp_name, 'fpr': fpr, 'tpr': tpr, 'auc': roc_auc})

# ---------- 4. BENDRŲ GRAFIKŲ GENERAVIMAS (ROC ir CM) ----------
print("\n" + "=" * 100)
print(" 4. GENERUOJAMI BENDRI GRAFIKAI (ROC IR CM) ".center(100, "="))

# --- 4.1 BENDRAS ROC GRAFIKAS ---
plt.figure(figsize=(10, 8))
colors = ['#1f77b4', '#ff7f0e']
for i, data in enumerate(roc_data_storage):
    plt.plot(data['fpr'], data['tpr'], color=colors[i % len(colors)], lw=3, label=f"{data['name']} (AUC = {data['auc']:.4f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Atsitiktinis')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('KNN ROC Kreivių Palyginimas')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
roc_filename = os.path.join(GRAFIKU_DIREKTORIJA, KNN_DIREKTORIJA, 'BENDRAS_ROC_Grafikas.png')
plt.savefig(roc_filename, dpi=300)
plt.close()
print(f"[OK] Sukurtas bendras ROC grafikas: {roc_filename}")

# --- 4.2 BENDRAS SUMAIŠYMO MATRICŲ GRAFIKAS ---
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

for i, data in enumerate(cm_data_storage):
    ax = axes[i]
    cm = data['cm']
    title = data['title']

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax, annot_kws={"size": 14})
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('Tikroji klasė', fontsize=10)
    ax.set_xlabel('Prognozuota klasė', fontsize=10)
    ax.set_xticklabels(['Normalus (0)', 'Aritmija (2)'])
    ax.set_yticklabels(['Normalus (0)', 'Aritmija (2)'])

plt.suptitle("KNN Sumaišymo Matricos Palyginimas", fontsize=16, y=1.02)
plt.tight_layout()
cm_filename = os.path.join(GRAFIKU_DIREKTORIJA, KNN_DIREKTORIJA, 'BENDRAS_Confusion_Matrix_Grid.png')
plt.savefig(cm_filename, dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] Sukurtas bendras CM grafikas (grid): {cm_filename}")


# ---------- 5. K PRIKLAUSOMYBES GRAFIKAI ----------
print("\n" + "=" * 100)
print(" 5. GENERUOJAMA METRIKU SUVESTINE (2x2 GRID) ".center(100, "="))

df_visos_metrikos = pd.DataFrame(visu_eksperimentu_duomenys)
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
custom_palette = {"Visi požymiai": "#1f77b4", "Optimalūs požymiai": "#ff7f0e"}

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, metric in enumerate(metrics_to_plot):
    ax = axes[i]

    sns.lineplot(
        data=df_visos_metrikos,
        x='k',
        y=metric,
        hue='Dataset',
        style='Dataset',
        markers=True,
        dashes=False,
        palette=custom_palette,
        linewidth=2.5,
        markersize=8,
        ax=ax,
        legend=(i == 0)
    )

    ax.set_title(f'{metric} priklausomybė nuo k', fontsize=12, fontweight='bold')
    ax.set_xlabel('k (Kaimynų skaičius)', fontsize=10)
    ax.set_ylabel(metric, fontsize=10)
    ax.set_xticks(range(1, 22, 2))
    ax.grid(True, linestyle='--', alpha=0.6)

plt.suptitle("KNN Metrikų Priklausomybė nuo k Reikšmės (Mokymo procesas)", fontsize=16, y=1.02)
plt.tight_layout()

combined_filename = os.path.join(GRAFIKU_DIREKTORIJA, KNN_DIREKTORIJA, 'KNN_Metriku_Suvestine_Grid.png')
plt.savefig(combined_filename, dpi=300, bbox_inches='tight')
plt.close()

print(f"[OK] Sukurtas bendras grafikas: {combined_filename}")
print(f"\n[INFO] Visi failai issaugoti aplanke: {os.path.join(GRAFIKU_DIREKTORIJA, KNN_DIREKTORIJA)}")
print("=" * 100)