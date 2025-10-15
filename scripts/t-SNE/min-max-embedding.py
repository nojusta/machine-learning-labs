import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

IN_PATH = "../../data/clean_data.csv"
LABEL = "NObeyesdad"
DROP = ["Height", "Weight", "Gender"]

label_names = {
    0: "Insufficient Weight",
    1: "Normal Weight",
    2: "Overweight Level I",
    3: "Overweight Level II",
    4: "Obesity Type I",
    5: "Obesity Type II",
    6: "Obesity Type III",
}

# --- įkėlimas ---
df = pd.read_csv(IN_PATH)
for c in DROP:
    if c in df.columns:
        df = df.drop(columns=c)

X_df = df.drop(columns=[LABEL]).copy()
y = df[LABEL].astype(int).values
mask = X_df.notna().all(axis=1)
X_df = X_df.loc[mask]
y = y[mask]

X_raw = X_df.values
X_mm = MinMaxScaler().fit_transform(X_raw)

os.makedirs("../../outputs/tsne", exist_ok=True)

def plot_tsne(X, y, perplexity, lr, max_iter, seed, tag):
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=lr,
        max_iter=max_iter,
        init="pca",
        early_exaggeration=12,
        random_state=seed,
    )
    Z = tsne.fit_transform(X)

    plt.figure(figsize=(7,6))
    classes = np.unique(y)
    colors = plt.cm.tab10(np.linspace(0, 1, max(7, len(classes))))
    for i, cls in enumerate(classes):
        m = (y == cls)
        label_str = f"{cls} – {label_names.get(cls, str(cls))}"
        plt.scatter(Z[m,0], Z[m,1], s=10, alpha=0.85, label=label_str, color=colors[i])
    plt.xlabel("t-SNE dimensija 1")
    plt.ylabel("t-SNE dimensija 2")
    plt.title(f"t-SNE ({tag}; perp={perplexity}, lr={lr}, iters={max_iter})")
    plt.legend(title="Klasės", markerscale=2, fontsize=8, loc="best")
    plt.tight_layout()
    out = f"../../outputs/tsne/tsne_{tag}_perp{perplexity}_lr{lr}_it{max_iter}_seed{seed}.png"
    plt.savefig(out, dpi=150)
    plt.show()
    print("Išsaugota:", out)
    return Z

# --- 1) Nenormuota (vienas bazinis) ---
plot_tsne(X_raw, y, perplexity=30, lr=200, max_iter=1000, seed=42, tag="raw")

# --- 2) Normuota (parametrų grid) ---
for perp in (30, 50, 80):
    for lr in (200, 400):
        plot_tsne(X_mm, y, perplexity=perp, lr=lr, max_iter=2000, seed=42, tag="minmax")

# --- 3) Stabilumas geriausiam deriniui (pvz., perp=50, lr=200) ---
for seed in (0, 42, 123):
    plot_tsne(X_mm, y, perplexity=50, lr=200, max_iter=2000, seed=seed, tag="minmax_best")
