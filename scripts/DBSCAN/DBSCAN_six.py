from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

# --- PARAMETRAI t-SNE ---
TSNE_PERPLEXITY = 40
TSNE_ITER = 600

# --- KELIAS IKI DUOMENŲ ---
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "normalized_minmax.csv"

FEATURES = ["Gender", "FCVC", "SMOKE", "CALC", "NCP", "CH2O"]

df = pd.read_csv(DATA_PATH)

df = df.drop(columns=["NObeyesdad"], errors="ignore")
X = df[FEATURES].to_numpy(dtype=float)



# --- PARAMETRŲ PAIEŠKA ---
eps_values = np.linspace(0.10, 2, 30)
min_samples_values = range(3, 25)

results = []

for eps in eps_values:
    for m in min_samples_values:
        model = DBSCAN(eps=eps, min_samples=m)
        labels = model.fit_predict(X)

        noise_ratio = (labels == -1).mean()
        mask = labels != -1

        if noise_ratio > 0.4 or len(set(labels[mask])) < 2:
            sil = -1
        else:
            sil = silhouette_score(X[mask], labels[mask])

        results.append((eps, m, sil))

results = pd.DataFrame(results, columns=["eps", "min_samples", "silhouette"])

best = results.loc[results["silhouette"].idxmax()]
BEST_EPS = best["eps"]
BEST_MIN = int(best["min_samples"])

print("\n=== Geriausi DBSCAN parametrai (B – 6 požymiai) ===")
print(f"eps={BEST_EPS:.3f}, min_samples={BEST_MIN}, silhouette={best['silhouette']:.4f}")

min_samples_for_knee = BEST_MIN  # naudok tuos pačius min_samples kaip radom silhouette paieškoje

nbrs = NearestNeighbors(n_neighbors=min_samples_for_knee).fit(X)
distances, indices = nbrs.kneighbors(X)
distances = np.sort(distances[:, -1])  # atstumai iki min_samples-to kaimyno

plt.figure(figsize=(7,4))
plt.plot(distances)
plt.title(f"k-Distance kreivė (min_samples={min_samples_for_knee})")
plt.xlabel("Taškai (surikiuoti)")
plt.ylabel("Atstumas iki min_samples-to kaimyno")
plt.grid(True)
plt.show()

# Automatinis alkūnės radimas (paprastas metodas: gradientas)
grad = np.gradient(distances)
knee_index = np.argmax(grad)   # kur kreivė staiga šauna aukštyn
KNEE_EPS = distances[knee_index]

print(f"\n=== EPS pagal alkūnės metodą (k-Distance) ===")
print(f"KNEE EPS ≈ {KNEE_EPS:.4f}\n")

# --- HEATMAP ---
plt.figure(figsize=(10,6))
sc = plt.scatter(results["eps"], results["min_samples"], c=results["silhouette"], cmap="viridis")
plt.colorbar(sc, label="Silhouette Score")
plt.xlabel("eps")
plt.ylabel("min_samples")
plt.title("DBSCAN parametrų analizė (6 požymiai)")
plt.show()

# --- GALUTINIS MODELIS ---
model = DBSCAN(eps=BEST_EPS, min_samples=BEST_MIN)
labels = model.fit_predict(X)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
noise = (labels == -1).mean()
mask = labels != -1
sil = silhouette_score(X[mask], labels[mask]) if n_clusters >= 2 else None

print(f"\n=== Geriausi DBSCAN parametrai (6 POZYMIAI) ===")
print(f"Optimalus eps: {BEST_EPS:.3f}")
print(f"Optimalus min_samples: {BEST_MIN}")
print(f"Didžiausias silhouette: {best['silhouette']:.4f}")

print("\n=== Galutiniai klasterizavimo rezultatai (6 POZYMIAI) ===")
print(f"Klasterių skaičius: {n_clusters}")
print(f"Triukšmo taškų dalis: {noise*100:.2f}%")
print(f"Silhouette: {sil:.4f}")

# --- t-SNE ---
Z = TSNE(
    n_components=2,
    perplexity=TSNE_PERPLEXITY,
    max_iter=TSNE_ITER,
    random_state=42
).fit_transform(X)

plt.figure(figsize=(8, 6))

unique_labels = sorted(set(labels))
cluster_labels = [cl for cl in unique_labels if cl != -1]

# Rankomis nustatomų spalvų rinkinys pirmiems 5 klasteriams
fixed_colors = [
    "#1f77b4",  # mėlyna
    "#d62728",  # raudona
    "#2ca02c",  # žalia
    "#ff7f0e",  # oranžinė
    "#9467bd"   # violetinė
]

color_map = {}

# 1) priskiriam spalvas pirmai 5 klasteriams (jei yra tiek)
for i, cl in enumerate(cluster_labels):
    if i < 5:
        color_map[cl] = fixed_colors[i]
    else:
        break

# 2) likusiems klasteriams – automatinės spalvos
if len(cluster_labels) > 5:
    remaining = cluster_labels[5:]
    auto_colors = plt.cm.tab20(np.linspace(0, 1, len(remaining)))
    for cl, col in zip(remaining, auto_colors):
        color_map[cl] = col

# 3) Braižymas
for cl in unique_labels:
    pts = Z[labels == cl]
    if cl == -1:
        col = "black"  # triukšmas juodai
        label = "Triukšmas"
    else:
        col = color_map[cl]
        label = f"Klasteris {cl}"
    plt.scatter(pts[:, 0], pts[:, 1], s=18, c=[col], label=label)

plt.title(f"DBSCAN (6 Požymiai)| eps={BEST_EPS:.3f}, min_samples={BEST_MIN}")

plt.xlabel("t-SNE dimensija 1")
plt.ylabel("t-SNE dimensija 2")

if len(cluster_labels) <= 10:
    plt.legend(loc="best", fontsize=8)

plt.show()