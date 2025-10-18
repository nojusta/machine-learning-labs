import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

IN_PATH = "../../data/clean_data.csv"
LABEL = "NObeyesdad"
KEEP_CLASSES = {4,5,6}
FEATURES = ["Gender","FCVC","SMOKE","CALC","NCP","CH2O"]

label_names = {
    4: "Obesity Type I",
    5: "Obesity Type II",
    6: "Obesity Type III",
}

df = pd.read_csv(IN_PATH)
df = df[df[LABEL].isin(KEEP_CLASSES)].copy()
X_df = df[FEATURES].copy()
y = df[LABEL].astype(int).values
mask = X_df.notna().all(axis=1)
X_df = X_df.loc[mask]
y = y[mask]

X_raw = X_df.values
X_mm = MinMaxScaler().fit_transform(X_raw)

os.makedirs("../../outputs/tsne", exist_ok=True)

def plot_tsne(X, y, perplexity, lr, max_iter, title_text, fname_tag):
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=lr,
        max_iter=max_iter,
        init="pca",
        early_exaggeration=12,
        random_state=42
    )
    Z = tsne.fit_transform(X)
    plt.figure(figsize=(7,6))
    classes = np.unique(y)
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    for i, cls in enumerate(classes):
        m = (y == cls)
        label_str = f"{cls} – {label_names.get(cls, str(cls))}"
        plt.scatter(Z[m,0], Z[m,1], s=12, alpha=0.9, label=label_str, color=colors[i])
    plt.xlabel("t-SNE dimensija 1")
    plt.ylabel("t-SNE dimensija 2")
    plt.title(title_text)
    plt.legend(title="Klasės", markerscale=2, fontsize=8, loc="best")
    plt.tight_layout()
    out = f"../../outputs/tsne/{fname_tag}.png"
    plt.savefig(out, dpi=150)
    plt.show()
    print("Išsaugota:", out)
    return Z

# 1) Nenormuota – 3 grafikai, be parametrų pavadinime
plot_tsne(
    X_raw, y,
    perplexity=30, lr=200, max_iter=1000,
    title_text="t-SNE (nenormuota)",
    fname_tag="tsne_raw_variantA_perp30_it1000"
)
plot_tsne(
    X_raw, y,
    perplexity=50, lr=200, max_iter=1000,
    title_text="t-SNE (nenormuota)",
    fname_tag="tsne_raw_variantB_perp50_it1000"
)
plot_tsne(
    X_raw, y,
    perplexity=80, lr=200, max_iter=1500,
    title_text="t-SNE (nenormuota)",
    fname_tag="tsne_raw_variantC_perp80_it1500"
)

# 2) Normuota – pirmiausia fokusas į n_iter, fiksuojam perplexity=50
for iters in (250, 500, 750, 1000):
    plot_tsne(
        X_mm, y,
        perplexity=50, lr=200, max_iter=iters,
        title_text=f"t-SNE (min–max; n_iter={iters})",
        fname_tag=f"tsne_minmax_iterScan_perp50_it{iters}"
    )
