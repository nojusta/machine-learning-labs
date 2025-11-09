import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

# 1. Duomenų nuskaitymas (jau normuoti, klasės 4–6 atrinktos)
df = pd.read_csv("../data/normalized_minmax_all.csv")

# 2. Pasirenkam požymius klasterizavimui (klasės nenaudojam)
X = df.drop(columns=["NObeyesdad"])

# 3. Ward hierarchinis klasterizavimas (linkage matrica)
Z_ward = linkage(X, method="ward")

# 4. Dendrograma (empirinis pjūvis)
plt.figure(figsize=(12, 6))
plt.title("Hierarchinė dendrograma (Ward)")
# truncate_mode='lastp' rodo tik paskutines jungtis, kad būtų aiškiau
dendrogram(
    Z_ward,
    truncate_mode="lastp",
    p=30,
    show_leaf_counts=True,
)
# čia pasirink aukštį pagal tai, kur matai aiškų šuolį
cut_height = 8
plt.axhline(y=cut_height, color="r", linestyle="--")
plt.xlabel("Sujungtų klasterių indeksas")
plt.ylabel("Atstumas")
plt.tight_layout()
plt.show()

# 5. Empirinis klasterių skaičius pagal pjūvio aukštį (neprivaloma, bet gali pasižiūrėt)
clusters_by_height = fcluster(Z_ward, t=cut_height, criterion="distance")
print("Klasterių skaičius pagal pjūvį h =", cut_height, ":",
      len(np.unique(clusters_by_height)))

# 6. Silhouette metodas k nuo 2 iki 10
range_n_clusters = range(2, 11)
silhouette_scores = []

for k in range_n_clusters:
    labels_k = fcluster(Z_ward, k, criterion="maxclust")
    score = silhouette_score(X, labels_k)
    silhouette_scores.append(score)
    print(f"k={k}: Silhouette={score:.3f}")

plt.figure(figsize=(8, 5))
plt.plot(range_n_clusters, silhouette_scores, marker="o")
plt.title("Silhouette metodas (Ward)")
plt.xlabel("Klasterių skaičius")
plt.ylabel("Silhouette reikšmė")
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. Pasirenkam galutinį k (pvz., pagal Silhouette maksimumą arba dendrogramą)
best_k = 3  # ČIA PASIKEISK pagal tai, ką nuspręsi (2 ar 3 ir t.t.)

final_labels = fcluster(Z_ward, best_k, criterion="maxclust")
df["cluster_ward_full"] = final_labels

print("Galutinis klasterių skaičius (Ward):", best_k)
print("Galutinis Silhouette:", silhouette_score(X, final_labels))

# 8. t-SNE vizualizacija (pilnai aibei, klasteriams pavaizduoti 2D)
tsne = TSNE(
    n_components=2,
    perplexity=40,
    max_iter=600,
    random_state=42,
)
X_tsne = tsne.fit_transform(X)

df["tsne_1"] = X_tsne[:, 0]
df["tsne_2"] = X_tsne[:, 1]

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    df["tsne_1"],
    df["tsne_2"],
    c=df["cluster_ward_full"],
    cmap="tab10",
    s=20,
)
plt.title(f"Hierarchinis klasterizavimas (Ward), k={best_k} – t-SNE projekcija")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.colorbar(scatter, label="Klasteris")
plt.tight_layout()
plt.show()
