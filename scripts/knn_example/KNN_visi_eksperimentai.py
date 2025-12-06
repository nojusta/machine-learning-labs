import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

# ---------- KONSTANTOS ----------
RANDOM_STATE = 42
DUOMENU_DIREKTORIJA = '../duomenys'
GRAFIKU_DIREKTORIJA = '../grafikai'
KNN_PAGRINDINE_DIR = 'KNN_eksperimentai'
JSON_DIREKTORIJA = '../JSON'
JSON_FAILAS = 'pozymiu_rinkiniai.json'

JSON_FAILAS_PATH = os.path.join(JSON_DIREKTORIJA, JSON_FAILAS)

# Sukuriame pagrindinę direktoriją rezultatams
BASE_OUTPUT_DIR = os.path.join(GRAFIKU_DIREKTORIJA, KNN_PAGRINDINE_DIR)
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# ---------- 1. ĮKELIAME DUOMENIS ----------
print("=" * 80)
print(" 1. DUOMENŲ ĮKĖLIMAS ".center(80, "="))
print("=" * 80)

try:
    df_mokymas = pd.read_csv(os.path.join(DUOMENU_DIREKTORIJA, 'mokymo_aibe.csv'), sep=';')
    df_validavimas = pd.read_csv(os.path.join(DUOMENU_DIREKTORIJA, 'validavimo_aibe.csv'), sep=';')
    df_testavimas = pd.read_csv(os.path.join(DUOMENU_DIREKTORIJA, 'testavimo_aibe.csv'), sep=';')
except FileNotFoundError:
    print(f"KLAIDA: Nerasti duomenų failai kataloge '{DUOMENU_DIREKTORIJA}'.")
    exit()

# ---------- 2. ĮKELIAME JSON KONFIGŪRACIJĄ ----------
try:
    with open(JSON_FAILAS_PATH, 'r', encoding='utf-8') as f:
        eksperimentai = json.load(f)
    print(f"✓ Rastas JSON failas. Įkelti {len(eksperimentai)} eksperimentų rinkiniai.")
except FileNotFoundError:
    print(f"KLAIDA: Nerastas '{JSON_FAILAS_PATH}'. Pirmiausia sugeneruokite jį.")
    exit()

# Kaupsime bendrus rezultatus ir ROC duomenis
visu_eksperimentu_rezultatai = []
roc_duomenys_bendrai = []

# ---------- 3. CIKLAS PER VISUS EKSPERIMENTUS ----------

print("\nPRADEDAMAS CIKLAS PER EKSPERIMENTUS...")

for eksp_pavadinimas, pozymiai in eksperimentai.items():
    # 3.1. Paruošiame direktoriją
    SAVE_DIR = os.path.join(BASE_OUTPUT_DIR, eksp_pavadinimas)
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 3.2. Atrenkame požymius
    try:
        X_mok = df_mokymas[pozymiai].values
        y_mok = df_mokymas['label'].values

        X_val = df_validavimas[pozymiai].values
        y_val = df_validavimas['label'].values

        X_test = df_testavimas[pozymiai].values
        y_test = df_testavimas['label'].values
    except KeyError as e:
        print(f"KLAIDA: Požymis {e} nerastas. Praleidžiamas '{eksp_pavadinimas}'.")
        continue

    # 3.3. Hiperparametrų parinkimas
    best_k = -1
    best_val_f1 = -1
    tuning_data = []

    for k in range(1, 22, 2):
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean', weights='uniform')
        knn.fit(X_mok, y_mok)
        y_val_pred = knn.predict(X_val)

        f1_val = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
        acc_val = accuracy_score(y_val, y_val_pred)

        tuning_data.append({'k': k, 'f1': f1_val, 'accuracy': acc_val})

        if f1_val > best_val_f1:
            best_val_f1 = f1_val
            best_k = k

    # Tuning grafikas
    td_df = pd.DataFrame(tuning_data)
    plt.figure(figsize=(8, 5))
    plt.plot(td_df['k'], td_df['f1'], marker='o', label='F1 Score')
    plt.plot(td_df['k'], td_df['accuracy'], marker='s', linestyle='--', label='Accuracy')
    plt.axvline(x=best_k, color='r', linestyle=':', label=f'Best k={best_k}')
    plt.title(f'Hiperparametrų parinkimas: {eksp_pavadinimas}')
    plt.xlabel('k')
    plt.ylabel('Metrika')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(SAVE_DIR, 'tuning_results.png'), dpi=300)
    plt.close()

    # 3.4. Galutinis modelis
    final_knn = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean', weights='uniform')
    final_knn.fit(X_mok, y_mok)
    y_test_pred = final_knn.predict(X_test)

    # 3.5. Metrikos
    acc = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

    visu_eksperimentu_rezultatai.append([
        eksp_pavadinimas, len(pozymiai), best_k, acc, prec, rec, f1
    ])

    print(f"  ✓ Atlikta: {eksp_pavadinimas} (k={best_k}, F1={f1:.4f})")

    # 3.6. Painiavos matrica
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'CM: {eksp_pavadinimas}\n(Acc: {acc:.4f})')
    plt.xlabel('Prognozuota')
    plt.ylabel('Tikra')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'confusion_matrix.png'), dpi=300)
    plt.close()

    # 3.7. ROC ir AUC kaupimas
    if hasattr(final_knn, "predict_proba"):
        y_proba = final_knn.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=2)
        roc_auc = auc(fpr, tpr)

        # Saugome individualų grafiką
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC={roc_auc:.4f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.title(f'ROC: {eksp_pavadinimas}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(SAVE_DIR, 'roc_curve.png'), dpi=300)
        plt.close()

        # Saugome duomenis bendram grafikui
        roc_duomenys_bendrai.append({
            'label': f"{eksp_pavadinimas} (AUC={roc_auc:.3f})",
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc
        })


# ---------- 4. REZULTATŲ LENTELĖ IR APIBENDRINIMAS ----------

# Rūšiuojame rezultatus
df_rezultatai = pd.DataFrame(
    visu_eksperimentu_rezultatai,
    columns=['Eksperimentas', 'Požymių sk.', 'Best k', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
)
df_rezultatai = df_rezultatai.sort_values(by='F1 Score', ascending=False).reset_index(drop=True)

print("\n" + "=" * 100)
print(" GALUTINIAI REZULTATAI (Surūšiuota pagal F1 Score) ".center(100, "="))
print("=" * 100)

print(tabulate(
    df_rezultatai, headers='keys', tablefmt='psql', floatfmt=".4f", showindex=False
))

# Bendras F1 stulpelių grafikas
plt.figure(figsize=(12, 8))
sns.barplot(x='F1 Score', y='Eksperimentas', data=df_rezultatai, palette='viridis')
plt.title('Visų eksperimentų F1 balų palyginimas')
plt.xlabel('F1 Score (Testavimo aibė)')
plt.xlim(0, 1.05)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE_OUTPUT_DIR, 'f1_score_summary.png'), dpi=300)
plt.close()

# Bendras ROC grafikas
if roc_duomenys_bendrai:
    print("\nGeneruojamas bendras ROC grafikas...")
    plt.figure(figsize=(10, 8))

    roc_duomenys_bendrai.sort(key=lambda x: x['auc'], reverse=True)

    for data in roc_duomenys_bendrai:
        plt.plot(data['fpr'], data['tpr'], lw=2, label=data['label'])

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Atsitiktinė prognozė')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Visų eksperimentų ROC kreivių palyginimas')
    plt.legend(loc="lower right", fontsize='small')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(BASE_OUTPUT_DIR, 'roc_curves_combined.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Bendras ROC grafikas išsaugotas: {save_path}")
else:
    print("Bendras ROC negeneruotas (trūksta duomenų).")

# ---------- 5. GERIAUSIŲ REZULTATŲ SUVESTINĖ (MODIFIKACIJA) ----------
print("\n" + "=" * 80)
print(" GERIAUSI RINKINIAI PAGAL KIEKVIENĄ METRIKĄ ".center(80, "="))
print("=" * 80)

# Randame indeksus su geriausiomis reikšmėmis
best_acc_idx = df_rezultatai['Accuracy'].idxmax()
best_prec_idx = df_rezultatai['Precision'].idxmax()
best_rec_idx = df_rezultatai['Recall'].idxmax()
best_f1_idx = df_rezultatai['F1 Score'].idxmax()

# Paimame atitinkamas eilutes
row_acc = df_rezultatai.loc[best_acc_idx]
row_prec = df_rezultatai.loc[best_prec_idx]
row_rec = df_rezultatai.loc[best_rec_idx]
row_f1 = df_rezultatai.loc[best_f1_idx]

# Spausdiname rezultatus
print(f"Didžiausias ACCURACY:   {row_acc['Eksperimentas']:<20} (Reikšmė: {row_acc['Accuracy']:.4f})")
print(f"Didžiausias PRECISION:  {row_prec['Eksperimentas']:<20} (Reikšmė: {row_prec['Precision']:.4f})")
print(f"Didžiausias RECALL:     {row_rec['Eksperimentas']:<20} (Reikšmė: {row_rec['Recall']:.4f})")
print(f"Didžiausias F1 SCORE:   {row_f1['Eksperimentas']:<20} (Reikšmė: {row_f1['F1 Score']:.4f})")

print("-" * 80)
print(f"Visi rezultatai išsaugoti aplanke: {BASE_OUTPUT_DIR}")