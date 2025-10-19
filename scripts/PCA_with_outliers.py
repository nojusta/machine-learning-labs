import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D

df = pd.read_csv("../data/outliers.csv")

features = ["Gender", "FCVC", "SMOKE", "CALC", "NCP", "CH2O"]
X = df[features]
y = df["NObeyesdad"]
outlier_type = df["outlier_type"]

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df["NObeyesdad"] = y
pca_df["outlier_type"] = outlier_type

class_colors = {
    4: "#F399FF",
    5: "#0080DB",
    6: "#48A348",
}

plt.figure(figsize=(8, 6))

for level, base_color in class_colors.items():
    subset = pca_df[pca_df["NObeyesdad"] == level]
    normal = subset[subset["outlier_type"] == 0]
    plt.scatter(normal["PC1"], normal["PC2"], color=base_color, edgecolor='none', alpha=0.8, s=55)
    inner = subset[subset["outlier_type"] == 1]
    plt.scatter(inner["PC1"], inner["PC2"], facecolor=base_color, edgecolor='black', linewidth=1.1, alpha=0.95, s=55)
    outer = subset[subset["outlier_type"] == 2]
    plt.scatter(outer["PC1"], outer["PC2"], facecolor=base_color, edgecolor='red', linewidth=1.3, alpha=0.95, s=55)

legend_elements = [
    Line2D([0], [0], marker='o', linestyle='None',
           markerfacecolor=class_colors[4], markeredgecolor='none',
           label='Nutukimo tipas I', markersize=9),
    Line2D([0], [0], marker='o', linestyle='None',
           markerfacecolor=class_colors[5], markeredgecolor='none',
           label='Nutukimo tipas II', markersize=9),
    Line2D([0], [0], marker='o', linestyle='None',
           markerfacecolor=class_colors[6], markeredgecolor='none',
           label='Nutukimo tipas III', markersize=9),
    Line2D([0], [0], marker='o', linestyle='None',
           markerfacecolor='white', markeredgecolor='black',
           label='Vidinė išskirtis', markersize=9, markeredgewidth=1.3),
    Line2D([0], [0], marker='o', linestyle='None',
           markerfacecolor='white', markeredgecolor='red',
           label='Išorinė išskirtis', markersize=9, markeredgewidth=1.3),
]

plt.legend(handles=legend_elements, title="Kategorijos", frameon=True)
plt.title("PCA", fontsize=13)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

print("\nPCA paaiškinta variacija:")
print(f"PC1: {pca.explained_variance_ratio_[0]:.3f}")
print(f"PC2: {pca.explained_variance_ratio_[1]:.3f}")
print(f"Bendra: {sum(pca.explained_variance_ratio_):.3f}")
