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
# DUOMENYS (su klase ir be klasės)
# =======================
df_full = pd.read_csv(DATA_PATH)                # su NObeyesdad – reikės palyginimui
y_raw = df_full["NObeyesdad"]                   # reali klasė (gali būti tekstinė)

# užsikoduoja klases į skaičius, kad būtų patogu piešti
y_codes, class_uniques = pd.factorize(y_raw)    # y_codes: 0..K-1
class_map = {i: cls for i, cls in enumerate(class_uniques)}

# DBSCAN'e klasės nenaudojamos
df = df_full.drop(columns=["NObeyesdad"], errors="ignore")
X = df.to_numpy(float)

# =======================
# 1. EPS nustatymas (k-distance knee)
# =======================
k0 = 5   # pradinis kaimynų skaičius k-distance kreivei
min_samples = 5  # PASIKEISK jei reikia

nbrs = NearestNeighbors(n_neighbors=k0).fit(X)
distances, _ = nbrs.kneighbors(X)
k_distances = np.sort(distances[:, -1])

# Randam "knee"
knee = KneeLocator(range(len(k_distances)), k_distances,
                   curve='convex', direction='increasing')
EPS = float(k_distances[knee.knee])

print(f"\n=== EPS pagal alkūnės metodą ===")
print(f"Rekomenduojamas eps ≈ {EPS:.4f}\n")

# ---- k-distance grafikas su linija ----
plt.figure(figsize=(8,5))
plt.plot(k_distances, label="k-distance kreivė")
plt.axhline(EPS, color='red', linestyle='--', label=f"EPS ≈ {EPS:.3f}")
plt.title("k-distance kreivė")
plt.xlabel("Taškai (surūšiuoti pagal atstumą)")
plt.ylabel(f"{k0}-to kaimyno atstumas")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / f"Knee_eps{EPS:.3f}.png", dpi=300)
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
# 3. t-SNE Vizualizacija (ta pati projekcija naudota visiems palyginimams)
# =======================
Z = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, max_iter=TSNE_ITER, random_state=42).fit_transform(X)

# Paprasta DBSCAN vizualizacija (kaip buvo)
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
if len(unique_labels) <= 15:
    plt.legend(fontsize=8)
plt.tight_layout()
fname_main = f"A_eps{EPS:.3f}_min{min_samples}.png"
plt.savefig(PLOT_DIR / fname_main, dpi=300, bbox_inches='tight')
print(f"\n✅ Paveikslas išsaugotas: {PLOT_DIR / fname_main}\n")
plt.show()

# =======================
# 4. Palyginimas su realiomis klasėmis
# =======================
# Confusion lentelė: klasteris x reali klasė
df_compare = pd.DataFrame({"real_code": y_codes, "cluster": labels})
confusion = pd.crosstab(df_compare["cluster"], df_compare["real_code"])
print("\n=== Klasterių palyginimas su realiomis klasėmis (kodais) ===")
print(confusion)

# Mapinam kiekvienam klasteriui dominuojančią klasę
major_class_by_cluster = (
    df_compare.groupby("cluster")["real_code"]
    .agg(lambda s: s.value_counts().idxmax())
    .to_dict()
)

# Ar konkretaus taško klasterio dominuojanti klasė sutampa su realia klase?
match = df_compare.apply(
    lambda r: (r["cluster"] != -1) and (r["real_code"] == major_class_by_cluster.get(r["cluster"], -999)),
    axis=1
).values  # bool masyvas

# Neatitikimų skaičius pagal realią klasę
mismatch_per_class = pd.Series(~match).groupby(y_codes).sum().astype(int)
print("\nNeatitikimai pagal klasę:")
for code, cnt in mismatch_per_class.items():
    print(f"  Klasė {code} ({class_map[code]}): {cnt} neatitinkančių objektų")

overall_acc = match.mean()*100
print(f"\nBendras atitikimas: {overall_acc:.1f}% (pilka=atitinka, raudona=neatitinka vizualizacijoje)")

# =======================
# 5. Trijų paveikslų palyginimas (kaip komandos nario)
#    1) t-SNE su klasėmis
#    2) t-SNE su DBSCAN klasteriais
#    3) Persidengimas: pilka – sutampa, raudona – nesutampa
# =======================
plt.figure(figsize=(16,5))

# 1) t-SNE su realiomis klasėmis
plt.subplot(1,3,1)
sc1 = plt.scatter(Z[:,0], Z[:,1], c=y_codes, cmap="tab10", s=14)
plt.title("t-SNE su klasėmis")
plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")

# 2) t-SNE su DBSCAN klasteriais
plt.subplot(1,3,2)
# triukšmas juodai, klasteriai – tab10
noise_mask = (labels == -1)
plt.scatter(Z[noise_mask,0], Z[noise_mask,1], c="black", s=14, label="Triukšmas")
plt.scatter(Z[~noise_mask,0], Z[~noise_mask,1], c=labels[~noise_mask], cmap="tab10", s=14)
plt.title("t-SNE su DBSCAN klasteriais")
plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")

# 3) Persidengimas
plt.subplot(1,3,3)
colors = np.where(match, "gray", "red")
plt.scatter(Z[:,0], Z[:,1], c=colors, s=14)
plt.title("t-SNE persidengimas (pilka=atitinka, raudona=neatitinka)")
plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")

plt.tight_layout()
fname_cmp = f"Compare_eps{EPS:.3f}_min{min_samples}.png"
plt.savefig(PLOT_DIR / fname_cmp, dpi=300, bbox_inches='tight')
print(f"\n✅ Palyginimo paveikslas išsaugotas: {PLOT_DIR / fname_cmp}\n")
plt.show()

# Papildomai – išsaugom confusion CSV ir žemėlapį klasės kodas -> pavadinimas
confusion_path = PLOT_DIR / f"Confusion_eps{EPS:.3f}_min{min_samples}.csv"
confusion.to_csv(confusion_path)
map_path = PLOT_DIR / "class_code_map.csv"
pd.Series(class_map).to_csv(map_path, header=["class_name"])
print(f"✅ Confusion lentelė: {confusion_path}")
print(f"✅ Klasės kodų žemėlapis: {map_path}")
