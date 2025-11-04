import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from matplotlib.lines import Line2D
from scipy.spatial.distance import pdist, squareform

# Klasės reikšmių aprašas
class_map = {
    0: "per_mažas_svoris",
    1: "normalus_svoris",
    2: "viršsvorio_lygis_1",
    3: "viršsvorio_lygis_2",
    4: "nutukimo_tipas_1",
    5: "nutukimo_tipas_2",
    6: "nutukimo_tipas_3"
}

# Spalvų paletė
palette = {
    4: "#F399FF",
    5: "#0080DB",
    6: "#48A348",
}

# Duomenų įkėlimas
df = pd.read_csv("../data/outliers.csv")

features = ["Gender", "FCVC", "SMOKE", "CALC", "NCP", "CH2O"]
X = df[features]
y = df["NObeyesdad"]
outlier_type = df["outlier_type"]

# MDS modelio parametrai
n_init_val = 10
max_iter_val = 1000

mds = MDS(
    n_components=2,
    random_state=42,
    dissimilarity="euclidean",
    n_init=n_init_val,
    max_iter=max_iter_val,
    n_jobs=8
)

X_mds = mds.fit_transform(X)

# === Normalizuotos streso reikšmės skaičiavimas ===
original_distances = pdist(X)  # atstumai pirminėje erdvėje
mds_distances = pdist(X_mds)   # atstumai MDS erdvėje
normalized_stress = np.sqrt(np.sum((original_distances - mds_distances) ** 2) /
                            np.sum(original_distances ** 2))

# Rezultatai į DataFrame
mds_df = pd.DataFrame(X_mds, columns=["Dim1", "Dim2"])
mds_df["NObeyesdad"] = y
mds_df["outlier_type"] = outlier_type

# Vizualizacija
plt.figure(figsize=(8, 6))

for level, base_color in palette.items():
    subset = mds_df[mds_df["NObeyesdad"] == level]

    normal = subset[subset["outlier_type"] == 0]
    plt.scatter(normal["Dim1"], normal["Dim2"],
                color=base_color, edgecolor='none', alpha=0.8, s=55)

    inner = subset[subset["outlier_type"] == 1]
    plt.scatter(inner["Dim1"], inner["Dim2"],
                facecolor=base_color, edgecolor='black', linewidth=1.1, alpha=0.95, s=55)

    outer = subset[subset["outlier_type"] == 2]
    plt.scatter(outer["Dim1"], outer["Dim2"],
                facecolor=base_color, edgecolor='red', linewidth=1.3, alpha=0.95, s=55)

# Legenda
legend_elements = [
    Line2D([0], [0], marker='o', linestyle='None',
           markerfacecolor=palette[4], markeredgecolor='none',
           label='Nutukimo tipas 1', markersize=9),
    Line2D([0], [0], marker='o', linestyle='None',
           markerfacecolor=palette[5], markeredgecolor='none',
           label='Nutukimo tipas 2', markersize=9),
    Line2D([0], [0], marker='o', linestyle='None',
           markerfacecolor=palette[6], markeredgecolor='none',
           label='Nutukimo tipas 3', markersize=9),
    Line2D([0], [0], marker='o', linestyle='None',
           markerfacecolor='white', markeredgecolor='black',
           label='Vidinė išskirtis', markersize=9, markeredgewidth=1.3),
    Line2D([0], [0], marker='o', linestyle='None',
           markerfacecolor='white', markeredgecolor='red',
           label='Išorinė išskirtis', markersize=9, markeredgewidth=1.3),
]

plt.legend(handles=legend_elements, title="Kategorijos", frameon=True)

plt.title(f"MDS (n_init={n_init_val}, max_iter={max_iter_val})", fontsize=13)
plt.xlabel("MDS dimensija 1")
plt.ylabel("MDS dimensija 2")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# Konsolės išvestis
print(f"🔹 Ne-normalizuotas stresas (sklearn): {mds.stress_:.2f}")
print(f"🔹 Normalizuotas stresas (0–1): {normalized_stress:.4f}")
