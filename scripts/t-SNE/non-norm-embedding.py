import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

IN_PATH = "../../data/clean_data.csv"
LABEL = "NObeyesdad"
DROP = ["Height", "Weight", "Gender"]

# Klasės mapping
label_names = {
    0: "Insufficient Weight",
    1: "Normal Weight",
    2: "Overweight Level I",
    3: "Overweight Level II",
    4: "Obesity Type I",
    5: "Obesity Type II",
    6: "Obesity Type III",
}

# ---- 1) Įkėlimas ir pasiruošimas
df = pd.read_csv(IN_PATH)

# numetam, jei dar likę
for c in DROP:
    if c in df.columns:
        df = df.drop(columns=c)

# atskiriam X ir y, išmetam eiles su NaN
X = df.drop(columns=[LABEL]).copy()
y = df[LABEL].copy()
mask = X.notna().all(axis=1) & y.notna()
X = X.loc[mask].values
y = y.loc[mask].values.astype(int)

# ---- 2) t-SNE (nenormuota)
tsne = TSNE(
    n_components=2,
    perplexity=80,
    learning_rate=200,
    max_iter=1000,
    init="pca",
    early_exaggeration=12,
    random_state=42
)
Z = tsne.fit_transform(X)

# ---- 3) Braižymas su pavadinimais
plt.figure(figsize=(7,6))
classes = np.unique(y)
colors = plt.cm.tab10(np.linspace(0, 1, max(7, len(classes))))

for i, cls in enumerate(classes):
    m = (y == cls)
    label_str = f"{cls} – {label_names.get(cls, str(cls))}"
    plt.scatter(Z[m,0], Z[m,1], s=10, alpha=0.85, label=label_str, color=colors[i])

plt.xlabel("t-SNE dimensija 1")
plt.ylabel("t-SNE dimensija 2")
plt.title("t-SNE (nenormuota pasirinkta aibė; perp=30, lr=200)")
plt.legend(title="Klasės", markerscale=2, fontsize=8, loc="best")
plt.tight_layout()

os.makedirs("../outputs/tsne", exist_ok=True)
out_path = "../outputs/tsne/tsne_raw_perp30_lr200.png"
plt.savefig(out_path, dpi=150)
plt.show()

print(f"Išsaugota: {out_path}")
