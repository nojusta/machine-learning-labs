import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.manifold import TSNE
from scipy.stats import mode
from matplotlib.lines import Line2D
from sklearn.metrics import accuracy_score

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

def cluster_stats(labels):
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique, counts))

def get_cluster_labels(Z, k):
    """Grąžina klasterių etiketes pagal 'maxclust' (k klasterių)."""
    return fcluster(Z, t=k, criterion="maxclust")


def print_cluster_descriptives(name, cluster_labels, class_labels):
    """
    Išveda klasterių aprašomąją statistiką:
    - dydžius
    - klasių (4/5/6) pasiskirstymą klasteriuose.
    """
    print(f"\n=== {name} ===")
    # Klasterių dydžiai
    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    print("Klasterių dydžiai (klasteris: objektų sk.):")
    for c, n in zip(unique_clusters, counts):
        print(f"  {c}: {n}")

    # Klasės pagal klasterius
    unique_classes = np.unique(class_labels)
    print("\nKlasės pagal klasterius (kiekis):")
    for c in unique_clusters:
        mask_cluster = (cluster_labels == c)
        cls_in_cluster = class_labels[mask_cluster]
        vals, cnts = np.unique(cls_in_cluster, return_counts=True)
        line = f"  Klasteris {c}: "
        parts = []
        for v, ct in zip(vals, cnts):
            parts.append(f"klasė {v}: {ct}")
        print(line + ", ".join(parts))

def map_clusters_to_classes(cluster_labels, true_labels):
    """
    Priskiria kiekvienam klasteriui tą klasę, kuri jame pasitaiko dažniausiai.
    Grąžina masyvą mapped_preds su tomis pačiomis dimensijomis kaip cluster_labels.
    """
    mapping = {}
    for cl in np.unique(cluster_labels):
        mask = (cluster_labels == cl)
        majority_class = mode(true_labels[mask], keepdims=True)[0][0]
        mapping[cl] = majority_class

    mapped_preds = np.array([mapping[c] for c in cluster_labels])
    return mapped_preds, mapping


def compute_mismatch_per_class(true_labels, mapped_preds):
    """
    Suskaičiuoja, kiek neatitikimų yra kiekvienoje klasėje.
    """
    mismatch_counts = {}
    classes = np.unique(true_labels)
    for cls in classes:
        mask = (true_labels == cls)
        mismatch_counts[int(cls)] = int(np.sum(true_labels[mask] != mapped_preds[mask]))
    return mismatch_counts


def plot_tsne_overlap(X_tsne2d, true_labels, mapped_preds, title_suffix=""):
    matches = (true_labels == mapped_preds)
    colors_overlap = np.where(matches, "gray", "red")

    unique_classes = np.unique(true_labels)

    plt.figure(figsize=(12, 4))

    # 1) t-SNE su tikromis klasėmis
    plt.subplot(1, 3, 1)
    for cls in unique_classes:
        m = (true_labels == cls)
        plt.scatter(
            X_tsne2d[m, 0],
            X_tsne2d[m, 1],
            s=15,
            label=f"Klasė {cls}"
        )
    plt.title("t-SNE su klasėmis")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(fontsize=8, loc="best")

    # 2) t-SNE su klasterizavimo rezultatais (po priskyrimo klasėms)
    plt.subplot(1, 3, 2)
    for cls in unique_classes:
        m = (mapped_preds == cls)
        plt.scatter(
            X_tsne2d[m, 0],
            X_tsne2d[m, 1],
            s=15,
        )
    plt.title("t-SNE su klasteriais (hierarchinis)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(fontsize=8, loc="best")

    # 3) Atitikimas / neatitikimas
    plt.subplot(1, 3, 3)
    plt.scatter(
        X_tsne2d[:, 0],
        X_tsne2d[:, 1],
        c=colors_overlap,
        s=15
    )
    plt.title(f"t-SNE persidengimas{title_suffix}")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    match_legend = Line2D(
        [0], [0],
        marker="o",
        color="w",
        markerfacecolor="gray",
        markersize=5,
        label="Atitinka klasę"
    )
    mismatch_legend = Line2D(
        [0], [0],
        marker="o",
        color="w",
        markerfacecolor="red",
        markersize=5,
        label="Neatitinka klasės"
    )
    plt.legend(handles=[match_legend, mismatch_legend],
               fontsize=8,
               loc="best")

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

# ---------- Klasterių palyginimas ir stabilumas (k=3, atrinkti 6 požymiai) ----------
df_outliers = pd.read_csv("../data/outliers.csv")

classes_k3 = df_outliers["NObeyesdad"].to_numpy()
outlier_type_k3 = df_outliers["outlier_type"].to_numpy()

labels_sel_k3 = get_cluster_labels(Z_sel, k=3)

rows = []
for c in sorted(np.unique(labels_sel_k3)):
    mask = (labels_sel_k3 == c)
    rows.append({
        "Klasteris": int(c),
        "Objektų sk.": int(mask.sum()),
        "Klasė 4": int(((classes_k3 == 4) & mask).sum()),
        "Klasė 5": int(((classes_k3 == 5) & mask).sum()),
        "Klasė 6": int(((classes_k3 == 6) & mask).sum()),
        "Vidinių išskirčių (1)": int(((outlier_type_k3 == 1) & mask).sum()),
        "Išorinių išskirčių (2)": int(((outlier_type_k3 == 2) & mask).sum()),
    })

df_k3_summary = pd.DataFrame(rows)
print("\n5.3.1 klasterių statistika (atrinkti 6 požymiai, k=3):")
print(df_k3_summary.to_string(index=False))

# ---------- Klasterių tikslumas ----------
true_classes = df_sel["NObeyesdad"].to_numpy()

k_compare = 3
clusters_tsne_k3 = fcluster(Z_tsne, t=k_compare, criterion="maxclust")

mapped_preds, cluster_to_class = map_clusters_to_classes(clusters_tsne_k3, true_classes)

accuracy = accuracy_score(true_classes, mapped_preds)
total = len(true_classes)
mismatch_total = int(np.sum(true_classes != mapped_preds))
mismatch_rate = mismatch_total / total * 100

print(f"\n=== Klasterizavimo tikslumas (t-SNE aibė, k=3) ===")
print(f"Tikslumas: {accuracy*100:.2f}%")
print(f"Neatitinkančių objektų: {mismatch_total} iš {total} ({mismatch_rate:.2f}%)")
print("Klasterių → klasių atvaizdavimas:", cluster_to_class)

mismatch_per_class = compute_mismatch_per_class(true_classes, mapped_preds)
print("\nNeatitikimai pagal klasę:")
for cls, cnt in mismatch_per_class.items():
    print(f"  Klasė {cls}: {cnt} neatitinkančių objektų")

plot_tsne_overlap(X_tsne2d, true_classes, mapped_preds,
                  title_suffix=f" (k={k_compare})")