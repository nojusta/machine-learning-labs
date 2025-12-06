import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold

# ---------- NUSTATYMAI ----------
DUOMENU_DIREKTORIJA = '../duomenys'
GRAFIKU_DIREKTORIJA = '../grafikai'
KNN_DIREKTORIJA = 'KNN'
CV_DIREKTORIJA = os.path.join(GRAFIKU_DIREKTORIJA, KNN_DIREKTORIJA, 'Cross_Validation')
JSON_DIREKTORIJA = '../JSON'
JSON_FAILAS = os.path.join(JSON_DIREKTORIJA, 'geriausias_rinkinys.json')

os.makedirs(CV_DIREKTORIJA, exist_ok=True)

BEST_K = 3

# ---------- 0. POŽYMIŲ NUSKAITYMAS ----------
print("=" * 100)
print(" KONFIGŪRACIJOS ĮKĖLIMAS ".center(100, "="))

try:
    with open(JSON_FAILAS, 'r', encoding='utf-8') as f:
        config = json.load(f)
        OPTIMALUS_POZYMIAI = config.get("GERIAUSIAS_MODELIS_6_POZYMIAI", [])

    if not OPTIMALUS_POZYMIAI:
        OPTIMALUS_POZYMIAI = ["Q_val", "R_val", "S_val", "Q_pos", "R_pos", "S_pos"]
        print("[INFO] JSON raktas tuščias, naudojami numatytieji požymiai.")
    else:
        print(f"[OK] Įkelti požymiai iš JSON: {len(OPTIMALUS_POZYMIAI)} vnt.")

except FileNotFoundError:
    OPTIMALUS_POZYMIAI = ["Q_val", "R_val", "S_val", "Q_pos", "R_pos", "S_pos"]
    print("[INFO] JSON nerastas, naudojami numatytieji požymiai.")

# ---------- 1. DUOMENŲ ĮKELIMAS ----------
print("-" * 100)
print(" VYKDOMAS KRYŽMINIS VALIDAVIMAS (10-Fold) ".center(100, " "))

try:
    df_mokymas = pd.read_csv(os.path.join(DUOMENU_DIREKTORIJA, 'mokymo_aibe.csv'), sep=';')
    df_validavimas = pd.read_csv(os.path.join(DUOMENU_DIREKTORIJA, 'validavimo_aibe.csv'), sep=';')
except FileNotFoundError:
    print(f"[KLAIDA] Nerasti duomenų failai aplanke '{DUOMENU_DIREKTORIJA}'.")
    exit()

try:
    X_full = np.concatenate([df_mokymas[OPTIMALUS_POZYMIAI].values, df_validavimas[OPTIMALUS_POZYMIAI].values])
    y_full = np.concatenate([df_mokymas['label'].values, df_validavimas['label'].values])
except KeyError as e:
    print(f"[KLAIDA] Trūksta stulpelių: {e}")
    exit()

print(f"Bendra aibė: {X_full.shape[0]} eilučių.")

# ---------- 2. SKAIČIAVIMAI ----------
cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
knn = KNeighborsClassifier(n_neighbors=BEST_K, metric='euclidean')

scoring_metrics = {
    'Accuracy': 'accuracy',
    'Precision': 'precision_weighted',
    'Recall': 'recall_weighted',
    'F1 Score': 'f1_weighted'
}

# Vykdome validavimą
results = cross_validate(knn, X_full, y_full, cv=cv_strategy, scoring=scoring_metrics)

# Konvertuojame į DataFrame
df_results = pd.DataFrame({
    'Accuracy': results['test_Accuracy'],
    'Precision': results['test_Precision'],
    'Recall': results['test_Recall'],
    'F1 Score': results['test_F1 Score']
})

# ---------- 3. DETALI STATISTIKA (Mean, Std, Var, Error) ----------
print("\n" + "=" * 100)
print(f" DETALI STATISTIKA (k={BEST_K}) ".center(100, "="))
print(f"{'METRIKA':<15} | {'VIDURKIS':<10} | {'STD (Nuokrypis)':<18} | {'VAR (Dispersija)':<18} | {'KLAIDA (Error)':<15}")
print("-" * 100)

for col in df_results.columns:
    values = df_results[col]

    mean_val = values.mean()
    std_val = values.std()
    var_val = std_val ** 2  # Dispersija yra standatinio nuokrypio kvadratas
    error_val = 1.0 - mean_val # Klaida yra 1 minus vidurkis (dažniausiai taikoma Accuracy, bet tinka ir kitiems)

    print(f"{col:<15} | {mean_val:.4f}    | {std_val:.4f}            | {var_val:.4f}            | {error_val:.4f}")

print("=" * 100)

# ---------- 4. VIZUALIZACIJA (JUODAI-BALTA) ----------
df_melted = df_results.melt(var_name='Metrika', value_name='Reikšmė')

plt.figure(figsize=(12, 7))

# Boxplot su juodais rėmeliais ir baltu vidumi
sns.boxplot(
    x='Metrika',
    y='Reikšmė',
    data=df_melted,
    color='white',
    width=0.5,
    linewidth=1.5,
    boxprops=dict(edgecolor='black'),
    whiskerprops=dict(color='black'),
    capprops=dict(color='black'),
    medianprops=dict(color='black', linewidth=2),
    flierprops=dict(marker='o', markerfacecolor='black', markersize=5, linestyle='none')
)

# Stripplot (taškai)
sns.stripplot(
    x='Metrika',
    y='Reikšmė',
    data=df_melted,
    color='black',
    size=5,
    jitter=True,
    alpha=0.6
)

plt.title(f'10-lanksčio kryžminio validavimo rezultatai\n(k={BEST_K}, Požymiai: Optimalūs)', fontsize=14, fontweight='bold', color='black')
plt.ylabel('Reikšmė (0.0 - 1.0)', fontsize=12, color='black')
plt.xlabel('', color='black')

plt.tick_params(axis='x', colors='black')
plt.tick_params(axis='y', colors='black')
plt.grid(True, axis='y', linestyle='--', alpha=0.5, color='gray')

# Ribos
y_min = df_melted['Reikšmė'].min()
plt.ylim(max(0, y_min - 0.02), 1.005)

failo_pav = os.path.join(CV_DIREKTORIJA, 'Kryzminis_Validavimas.png')
plt.savefig(failo_pav, dpi=300)
plt.close()

print(f"\n[OK] Juodai-baltas grafikas išsaugotas: {failo_pav}")
print("=" * 100)