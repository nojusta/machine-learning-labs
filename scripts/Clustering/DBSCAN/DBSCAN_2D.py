from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

# =======================
# PARAMETRAI
# =======================
TSNE_PERPLEXITY = 40
TSNE_ITER = 600
MIN_S = 10   # <<< ČIA PASIRENKAM min_samples

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "normalized_minmax.csv"   # PILNA AIBĖ

# Saugojimo vieta
PLOT_DIR = ROOT / "scripts" / "DBSCAN" / "2D_Rez"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# =======================
# 1. Duomenys
# =======================
df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["NObeyesdad"], errors="ignore")
X = df.to_numpy(float)

# =======================
# 2. t-SNE -> 2D
# =======================
print("Atliekamas t-SNE skaičiavimas (gali užtrukti)...")
Z = TSNE(
    n_components=2,
    perplexity=TSNE_PERPLEXITY,
    max_iter=TSNE_ITER,
    random_state=42
).fit_transform(X)

# =======================
# 3. EPS parinkimas (knee, kaip A ir B)
# =======================
k0 = 5
nbrs = NearestNeighbors(n_neighbors=k0).fit(Z)
distances, _ = nbrs.kneighbors(Z)
k_distances = np.sort(distances[:, -1])

knee = KneeLocator(range(len(k_distances)), k_distances, curve='convex', direction='increasing')
EPS = float(k_distances[knee.knee])

print(f"\nEPS ≈ {EPS:.4f} (iš 2D projekcijos)")

# Grafikas
plt.figure(figsize=(8,5))
plt.plot(k_distances)
plt.axhline(EPS, color='red', linestyle='--', label=f"EPS ≈ {EPS:.3f}")
plt.title("k-distance kreivė (2D projekcija)")
plt.xlabel("Taškai (surūšiuoti pagal atstumą)")
plt.ylabel(f"{k0}-to kaimyno atstumas")
plt.legend()
plt.grid(True)
plt.show()

# =======================
# 4. DBSCAN su VIENU min_samples
# =======================
model = DBSCAN(eps=EPS, min_samples=MIN_S)
labels = model.fit_predict(Z)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
noise_ratio = (labels == -1).mean() * 100
mask = labels != -1
sil = silhouette_score(Z[mask], labels[mask]) if n_clusters >= 2 else None

print(f"\n--- Rezultatai (min_samples={MIN_S}) ---")
print(f"Klasterių skaičius: {n_clusters}")
print(f"Triukšmo dalis: {noise_ratio:.1f}%")
print(f"Silhouette: {sil:.4f}" if sil else "Silhouette neįmanomas")

# =======================
# 5. Vizualizacija
# =======================
plt.figure(figsize=(8,6))
for cl in sorted(set(labels)):
    pts = Z[labels == cl]
    color = "black" if cl == -1 else None
    label = "Triukšmas" if cl == -1 else f"Klasteris {cl}"
    plt.scatter(pts[:,0], pts[:,1], s=18, c=color, label=label)

plt.title(f"DBSCAN 2D (eps={EPS:.3f}, min_samples={MIN_S}) | kl={n_clusters}, noise={noise_ratio:.1f}%")
plt.xlabel("t-SNE dimensija 1")
plt.ylabel("t-SNE dimensija 2")
plt.legend(fontsize=8)

file = f"C_2D_eps{EPS:.3f}_min{MIN_S}.png"
path = PLOT_DIR / file
plt.savefig(path, dpi=300, bbox_inches='tight')
plt.show()

print(f"Išsaugota: {path}")

# =======================
# 6. Klasterių interpretacija
# =======================
df_clustered = df.copy()
df_clustered["cluster"] = labels

cluster_means = df_clustered.groupby("cluster").mean()
print("\n=== Klasterių požymių vidurkiai ===")
print(cluster_means.round(3))
