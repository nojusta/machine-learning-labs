from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

# =======================
# PARAMETRAI
# =======================
TSNE_PERPLEXITY = 40
TSNE_ITER = 600

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "normalized_minmax.csv"

# Kur saugoti paveikslus
PLOT_DIR = ROOT / "scripts" / "DBSCAN" / "6_Poz_Rez"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# =======================
# DUOMENYS
# =======================
df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["NObeyesdad"], errors="ignore")
X = df.to_numpy(float)

# =======================
# 1. EPS nustatymas (k-distance knee)
# =======================
k0 = 5  # K-dėl alkūnės metodo
min_samples = 5

nbrs = NearestNeighbors(n_neighbors=k0).fit(X)
distances, _ = nbrs.kneighbors(X)
k_distances = np.sort(distances[:, -1])

# Randam alkūnę
knee = KneeLocator(range(len(k_distances)), k_distances,
                   curve='convex', direction='increasing')
EPS = float(k_distances[knee.knee])

print(f"\n=== EPS pagal alkūnės metodą ===")
print(f"Rekomenduojamas eps ≈ {EPS:.4f}\n")

# ---- K-distance grafikas su linija ----
plt.figure(figsize=(8,5))
plt.plot(k_distances, label="k-distance kreivė")
plt.axhline(EPS, color='red', linestyle='--', label=f"EPS ≈ {EPS:.3f}")
plt.title(f"k-distance kreivė")
plt.xlabel("Taškai ")
plt.ylabel(f"{k0}-to kaimyno atstumas")
plt.legend()
plt.grid(True)
plt.show()
# =======================
# 2. DBSCAN
# =======================
model = DBSCAN(eps=EPS, min_samples=min_samples)
labels = model.fit_predict(X)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
noise_ratio = (labels == -1).mean() * 100

mask = labels != -1
sil = silhouette_score(X[mask], labels[mask]) if n_clusters >= 2 else None

print("\n=== DBSCAN rezultatai (A variantas) ===")
print(f"eps = {EPS:.4f}")
print(f"min_samples = {min_samples}")
print(f"Klasterių skaičius: {n_clusters}")
print(f"Triukšmo dalis: {noise_ratio:.2f}%")
print(f"Silhouette: {sil:.4f}" if sil else "Silhouette: negalimas (per mažai klasterių)")


# =======================
# 3. t-SNE Vizualizacija
# =======================
Z = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, max_iter=TSNE_ITER, random_state=42).fit_transform(X)

plt.figure(figsize=(8, 6))
unique_labels = sorted(set(labels))

for cl in unique_labels:
    pts = Z[labels == cl]
    if cl == -1:
        plt.scatter(pts[:,0], pts[:,1], s=18, c="black", label="Triukšmas")
    else:
        plt.scatter(pts[:,0], pts[:,1], s=18, label=f"Klasteris {cl}")

plt.title(f"DBSCAN (6 Požymiai) (eps={EPS:.3f}, min_samples={min_samples}) | kl={n_clusters}, noise={noise_ratio:.1f}%")
plt.xlabel("t-SNE dimensija 1")
plt.ylabel("t-SNE dimensija 2")
plt.legend(fontsize=8)

# IŠSAUGOME
plt.legend(fontsize=8)

filename = f"A_eps{EPS:.3f}_min{min_samples}.png"
save_path = PLOT_DIR / filename

plt.savefig(save_path, dpi=300, bbox_inches='tight')  # <--- SAUGOJAM PIRMIAU
print(f"\n✅ Paveikslas išsaugotas: {save_path}\n")

plt.show()


# ======================================================================
# 5. Klasterių aprašomoji statistika su išskirtimis
# ======================================================================

# Įkeliam failą su išskirtimis
OUT_PATH = ROOT / "data" / "outliers.csv"
df_out = pd.read_csv(OUT_PATH)

# Paliekame tik klases 4, 5, 6 (pašaliname 0-3 klases, nes jų neturi būti tyrime)
df_out = df_out[df_out["NObeyesdad"].isin([4, 5, 6])].reset_index(drop=True)

# Paverčiame klases į kodus 0, 1, 2
df_out["class_code"] = df_out["NObeyesdad"].map({4:0, 5:1, 6:2})

# Pridedame DBSCAN klasterius
df_out["cluster"] = labels

# Lentelė: kiek įrašų kiekviename klasteryje
total_counts = df_out.groupby("cluster").size().rename("Iš viso įrašų")

# Kiekvienos klasės pasiskirstymas po klasterius
class_counts = df_out.groupby(["cluster", "class_code"]).size().unstack(fill_value=0)
class_counts.columns = ["0 klasės įrašų", "1 klasės įrašų", "2 klasės įrašų"]

# Išskirtys
outlier_counts = df_out.groupby(["cluster", "outlier_type"]).size().unstack(fill_value=0)
outlier_counts.columns = ["Įrašų be išskirčių", "Mild išskirčių", "Extreme išskirčių"]

# Sujungiame į vieną lentelę
summary_table = pd.concat([total_counts, class_counts, outlier_counts], axis=1)

print("\n=== DBSCAN klasterių aprašomoji statistika su išskirtimis ===")
print(summary_table)
