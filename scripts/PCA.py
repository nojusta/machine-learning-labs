import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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

# PCA su NORMUOTAIS duomenimis
print("PCA su NORMUOTAIS duomenimis")
df_norm = pd.read_csv('../data/normalized_minmax.csv')

features_norm = df_norm.select_dtypes(include=['float64', 'int64']).drop(columns=['NObeyesdad'], errors='ignore')

if 'Gender' in features_norm.columns:
    features_norm = features_norm.drop(columns=['Gender'])
    print("Pašalintas stulpelis: Gender")

labels_norm = df_norm['NObeyesdad'].map(class_map)

pca_norm = PCA(n_components=2, whiten=True)
pca_result_norm = pca_norm.fit_transform(features_norm)

pca_df_norm = pd.DataFrame(pca_result_norm, columns=['PC1', 'PC2'])
pca_df_norm['Kategorija'] = labels_norm

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(
    data=pca_df_norm,
    x='PC1', y='PC2',
    hue='Kategorija',
    palette=palette,
    alpha=0.85, s=60
)
plt.title("PCA (su normuotais duomenimis)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Kategorija", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)

print("\nPCA paaiškinta variacija (NORMUOTA)")
print(f"Pirmoji komponentė (PC1): {pca_norm.explained_variance_ratio_[0]:.3f}")
print(f"Antroji komponentė (PC2): {pca_norm.explained_variance_ratio_[1]:.3f}")
print(f"Bendra paaiškinta variacija: {sum(pca_norm.explained_variance_ratio_):.2f}")

loadings_norm = pd.DataFrame(
    pca_norm.components_.T,
    columns=['PC1', 'PC2'],
    index=features_norm.columns
)
importance_norm = loadings_norm.abs().sum(axis=1).sort_values(ascending=False)

print("\nPožymių įtakos lentelė (normuoti duomenys):\n")
print(loadings_norm.round(6))
print("\nBendra svarba (|PC1| + |PC2|):\n")
print(importance_norm.round(6))

# PCA su NENORMUOTAIS duomenimis
print("\n\nPCA su NENORMUOTAIS duomenimis")
df_raw = pd.read_csv('../data/full_clean_with_hw.csv')

for col in ['Gender', 'Height', 'Weight']:
    if col in df_raw.columns:
        df_raw = df_raw.drop(columns=[col])
        print(f"Pašalintas stulpelis: {col}")

features_raw = df_raw.select_dtypes(include=['float64', 'int64']).drop(columns=['NObeyesdad'], errors='ignore')
labels_raw = df_raw['NObeyesdad'].map(class_map)

pca_raw = PCA(n_components=2, whiten=True)
pca_result_raw = pca_raw.fit_transform(features_raw)

pca_df_raw = pd.DataFrame(pca_result_raw, columns=['PC1', 'PC2'])
pca_df_raw['Kategorija'] = labels_raw

plt.subplot(1, 2, 2)
sns.scatterplot(
    data=pca_df_raw,
    x='PC1', y='PC2',
    hue='Kategorija',
    palette=palette,
    alpha=0.85, s=60
)
plt.title("PCA (be normavimo)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

print("\nPCA paaiškinta variacija (BE NORMAVIMO)")
print(f"Pirmoji komponentė (PC1): {pca_raw.explained_variance_ratio_[0]:.3f}")
print(f"Antroji komponentė (PC2): {pca_raw.explained_variance_ratio_[1]:.3f}")
print(f"Bendra paaiškinta variacija: {sum(pca_raw.explained_variance_ratio_):.2f}")

loadings_raw = pd.DataFrame(
    pca_raw.components_.T,
    columns=['PC1', 'PC2'],
    index=features_raw.columns
)
importance_raw = loadings_raw.abs().sum(axis=1).sort_values(ascending=False)

print("\nPožymių įtakos lentelė (nenormuoti duomenys):\n")
print(loadings_raw.round(6))
print("\nBendra svarba (|PC1| + |PC2|):\n")
print(importance_raw.round(6))
