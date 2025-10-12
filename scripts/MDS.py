import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Bendri nustatymai
class_map = {
    0: "per_mažas_svoris",
    1: "normalus_svoris",
    2: "viršsvorio_lygis_1",
    3: "viršsvorio_lygis_2",
    4: "nutukimo_tipas_1",
    5: "nutukimo_tipas_2",
    6: "nutukimo_tipas_3"
}

palette = sns.color_palette("RdYlGn_r", len(class_map))

# === 1️⃣ MDS su NORMUOTAIS duomenimis ===
print("=== MDS su NORMUOTAIS duomenimis ===")
df_norm = pd.read_csv('../data/normalized_minmax.csv')

features_norm = df_norm.select_dtypes(include=['float64', 'int64']).drop(columns=['NObeyesdad'], errors='ignore')

if 'Gender' in features_norm.columns:
    features_norm = features_norm.drop(columns=['Gender'])
    print("🗑️ Pašalintas stulpelis: Gender")

labels_norm = df_norm['NObeyesdad'].map(class_map)

# StandardScaler (tinkamesnis MDS nei MinMax)
scaler = StandardScaler()
features_norm_scaled = scaler.fit_transform(features_norm)

# MDS modelis (2D)
mds_norm = MDS(n_components=2, random_state=42, dissimilarity='euclidean', n_init=4, max_iter=500)
mds_result_norm = mds_norm.fit_transform(features_norm_scaled)

# Rezultatas į DataFrame
mds_df_norm = pd.DataFrame(mds_result_norm, columns=['Dim1', 'Dim2'])
mds_df_norm['Kategorija'] = labels_norm

# Vizualizacija
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(
    data=mds_df_norm,
    x='Dim1', y='Dim2',
    hue='Kategorija',
    palette=palette,
    alpha=0.85, s=60
)
plt.title("MDS (su normuotais duomenimis)")
plt.xlabel("Dimensija 1")
plt.ylabel("Dimensija 2")
plt.legend(title="Kategorija", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)

print(f"\n--- MDS (normuoti duomenys) ---")
print(f"Stresas (distortion): {mds_norm.stress_:.2f}")

# === 2️⃣ MDS su NENORMUOTAIS duomenimis ===
print("\n=== MDS su NENORMUOTAIS duomenimis ===")
df_raw = pd.read_csv('../data/full_clean_with_hw.csv')

for col in ['Gender', 'Height', 'Weight']:
    if col in df_raw.columns:
        df_raw = df_raw.drop(columns=[col])
        print(f"🗑️ Pašalintas stulpelis: {col}")

features_raw = df_raw.select_dtypes(include=['float64', 'int64']).drop(columns=['NObeyesdad'], errors='ignore')
labels_raw = df_raw['NObeyesdad'].map(class_map)

# MDS modelis (be mastelio keitimo)
mds_raw = MDS(n_components=2, random_state=42, dissimilarity='euclidean', n_init=4, max_iter=500)
mds_result_raw = mds_raw.fit_transform(features_raw)

# Rezultatas į DataFrame
mds_df_raw = pd.DataFrame(mds_result_raw, columns=['Dim1', 'Dim2'])
mds_df_raw['Kategorija'] = labels_raw

# Vizualizacija
plt.subplot(1, 2, 2)
sns.scatterplot(
    data=mds_df_raw,
    x='Dim1', y='Dim2',
    hue='Kategorija',
    palette=palette,
    alpha=0.85, s=60
)
plt.title("MDS (be normavimo)")
plt.xlabel("Dimensija 1")
plt.ylabel("Dimensija 2")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

print(f"\n--- MDS (be normavimo) ---")
print(f"Stresas (distortion): {mds_raw.stress_:.2f}")

# === 3️⃣ Palyginimas pagal stresą ===
stress_data = pd.DataFrame({
    'Būsena': ['Normuoti', 'Be normavimo'],
    'Stresas': [mds_norm.stress_, mds_raw.stress_]
})

plt.figure(figsize=(5, 4))
sns.barplot(data=stress_data, x='Būsena', y='Stresas', palette=['teal', 'orange'])
plt.title('MDS streso palyginimas (mažesnis = geresnis)')
plt.xlabel('')
plt.ylabel('Streso reikšmė')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
