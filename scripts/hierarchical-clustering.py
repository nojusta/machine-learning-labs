import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.manifold import TSNE

# ---------- 1. Duomenų nuskaitymas ----------

# Pilna normuota aibė (16 požymių, be klasės)
df_full = pd.read_csv("../data/normalized_minmax_all.csv")
X_full = df_full.drop(columns=["NObeyesdad"])

# Atrinkti 6 požymiai (be klasės)
df_sel = pd.read_csv("../data/normalized_minmax.csv")
X_sel = df_sel.drop(columns=["NObeyesdad"])

# t-SNE 2D iš tų pačių 6 požymių
tsne_for_dataset = TSNE(
    n_components=2,
    perplexity=40,
    max_iter=600,
    random_state=42,
)
X_tsne2d = tsne_for_dataset.fit_transform(X_sel)
df_tsne = pd.DataFrame(X_tsne2d, columns=["tsne_1", "tsne_2"])

# ---------- 2. Pagalbinės funkcijos ----------

def cut_height_for_k(Z, k):
    """
    Apskaičiuoja pjūvio aukštį t taip, kad fcluster(..., t, 'distance')
    duotų k klasterių.
    """
    n = Z.shape[0] + 1
    if k < 2 or k > n:
        raise ValueError("k turi būti tarp 2 ir n")

    i = n - k

    if i == 0:
        t = Z[0, 2] * 0.5
    elif i >= Z.shape[0]:
        t = Z[-1, 2] + 1.0
    else:
        t = (Z[i - 1, 2] + Z[i, 2]) / 2.0

    return t

def plot_dendrogram_with_k(X, k, title, figsize=(10, 5)):
    """
    Sudaro Ward dendrogramą ir nupiešia pjūvio liniją taip,
    kad būtų k klasterių.
    """
    Z = linkage(X, method="ward")  # Ward + Euclidean
    t = cut_height_for_k(Z, k)

    plt.figure(figsize=figsize)
    plt.title(title)
    dendrogram(Z)
    plt.axhline(y=t, color="r", linestyle="--")
    plt.xlabel("Objektai / klasteriai")
    plt.ylabel("Atstumas")
    plt.tight_layout()
    plt.show()

    return Z, t


def plot_tsne_clusters_grid(X_2d, Z, base_k, title_prefix):
    """
    T-SNE 2D vizualizacija su 3 skirtingais k:
    jei base_k == 2 → k={2,3,4}
    kitaip → k={base_k-1, base_k, base_k+1}
    """
    if base_k == 2:
        ks = [2, 3, 4]
    else:
        ks = [base_k - 1, base_k, base_k + 1]

    plt.figure(figsize=(5 * len(ks), 4))

    for i, k in enumerate(ks, start=1):
        labels_k = fcluster(Z, k, criterion="maxclust")

        plt.subplot(1, len(ks), i)
        scatter = plt.scatter(
            X_2d[:, 0],
            X_2d[:, 1],
            c=labels_k,
            cmap="tab10",
            s=15,
        )
        plt.title(f"{title_prefix}, k={k}")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")

    plt.tight_layout()
    plt.show()

# ---------- 3. Dendrogramos ir k parinkimas ----------
k_full = 2
k_sel = 2
k_tsne = 2

Z_full, t_full = plot_dendrogram_with_k(
    X_full, k_full, "Dendrograma – pilna normuota aibė (16 požymių)"
)

Z_sel, t_sel = plot_dendrogram_with_k(
    X_sel, k_sel, "Dendrograma – atrinkti 6 požymiai"
)

Z_tsne, t_tsne = plot_dendrogram_with_k(
    df_tsne.values, k_tsne, "Dendrograma – t-SNE sumažinta aibė (2D)"
)

# ---------- 4. t-SNE vizualizacijos su k, k+1, k+2 ----------

tsne_full_vis = TSNE(
    n_components=2,
    perplexity=40,
    max_iter=600,
    random_state=42,
)
X_full_2d = tsne_full_vis.fit_transform(X_full)

plot_tsne_clusters_grid(
    X_full_2d,
    Z_full,
    k_full,
    "Pilna aibė (16 požymių), Ward"
)

tsne_sel_vis = TSNE(
    n_components=2,
    perplexity=40,
    max_iter=600,
    random_state=42,
)
X_sel_2d = tsne_sel_vis.fit_transform(X_sel)

plot_tsne_clusters_grid(
    X_sel_2d,
    Z_sel,
    k_sel,
    "Atrinkti 6 požymiai, Ward"
)

plot_tsne_clusters_grid(
    X_tsne2d,
    Z_tsne,
    k_tsne,
    "t-SNE 2D aibė, Ward"
)
