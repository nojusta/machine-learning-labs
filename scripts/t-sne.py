import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

NON_NORM_PATH = "../data/non-norm_outliers.csv"
NORM_PATH = "../data/outliers.csv"
LABEL = "NObeyesdad"
KEEP_CLASSES = {4, 5, 6}
FEATURES = ["Gender","FCVC","SMOKE","CALC","NCP","CH2O"]

label_names = {
    4: "Nutukimo tipas I",
    5: "Nutukimo tipas II",
    6: "Nutukimo tipas III",
}

class_colors = {
    4: "#F399FF",
    5: "#0080DB",
    6: "#48A348",
}

os.makedirs("../outputs/tsne", exist_ok=True)

def load_xy(path):
    df0 = pd.read_csv(path)
    total_rows = len(df0)
    df = df0[df0[LABEL].isin(KEEP_CLASSES)].copy()
    rows_after_class = len(df)

    if "outlier_type" not in df.columns:
        raise ValueError("Trūksta 'outlier_type' stulpelio (turi būti 0/1/2).")

    feat = df[FEATURES]
    mask_valid = feat.notna().all(axis=1)
    rows_after_nan = int(mask_valid.sum())

    X = feat.loc[mask_valid].to_numpy(dtype=float)
    y = df.loc[mask_valid, LABEL].astype(int).to_numpy()
    outlier_type = df.loc[mask_valid, "outlier_type"].astype(int).to_numpy()

    print(f"[{os.path.basename(path)}] eilutės: {total_rows} | po klasių filtro: {rows_after_class} | po NaN drop (TSNE naudoja): {rows_after_nan}")
    print(f"[{os.path.basename(path)}] vidinių (1): {(df['outlier_type']==1).sum()} | išorinių (2): {(df['outlier_type']==2).sum()} | iš viso outlierių: {(df['outlier_type']>0).sum()}")
    print(f"[TSNE naudojam subset] vidinių (1): {(outlier_type==1).sum()} | išorinių (2): {(outlier_type==2).sum()} | iš viso: {(outlier_type>0).sum()} / {len(outlier_type)}\n")

    return X, y, outlier_type

def plot_tsne(X, y, outlier_type, perplexity, lr, max_iter, title_text, fname_tag):
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

    normal = (outlier_type == 0)
    for cls in np.unique(y):
        m = (y == cls) & normal
        plt.scatter(
            Z[m,0], Z[m,1],
            color=class_colors.get(cls, "gray"),
            edgecolor='none',
            alpha=0.8, s=55,
            label=f"{cls} – {label_names.get(cls, str(cls))}"
        )

    # Vidinės – juodas apvadas, facecolor = klasės spalva
    inner = (outlier_type == 1)
    if inner.any():
        for cls in np.unique(y):
            m = (y == cls) & inner
            if m.any():
                plt.scatter(
                    Z[m,0], Z[m,1],
                    facecolor=class_colors.get(cls, "gray"),
                    edgecolor='black', linewidth=1.1,
                    alpha=0.95, s=55
                )

    # Išorinės – raudonas apvadas, facecolor = klasės spalva
    outer = (outlier_type == 2)
    if outer.any():
        for cls in np.unique(y):
            m = (y == cls) & outer
            if m.any():
                plt.scatter(
                    Z[m,0], Z[m,1],
                    facecolor=class_colors.get(cls, "gray"),
                    edgecolor='red', linewidth=1.3,
                    alpha=0.95, s=55
                )

    inner_proxy = plt.Line2D([0],[0], marker='o', color='w',
                             markerfacecolor='none', markeredgecolor='black',
                             markersize=6, linewidth=0, label='Vidinės išskirtys')
    outer_proxy = plt.Line2D([0],[0], marker='o', color='w',
                             markerfacecolor='none', markeredgecolor='red',
                             markersize=6, linewidth=0, label='Išorinės išskirtys')

    handles, labels = plt.gca().get_legend_handles_labels()
    handles += [inner_proxy, outer_proxy]
    labels += ['Vidinė išskirtis', 'Išorinė išskirtis']

    plt.xlabel("t-SNE dimensija 1")
    plt.ylabel("t-SNE dimensija 2")
    plt.title(title_text)
    plt.legend(handles, labels, title="Klasės / išskirtys", markerscale=1.2, fontsize=8, loc="best")
    plt.tight_layout()
    out_path = f"../outputs/tsne/{fname_tag}.png"
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Išsaugota: {out_path}\n")
    return Z

# 1) Nenormuota – keli grafikai su skirtingu perplexity
X_raw, y_raw, out_raw = load_xy(NON_NORM_PATH)
plot_tsne(X_raw, y_raw, out_raw, perplexity=30, lr=200, max_iter=500,
          title_text="t-SNE (nenormuota)", fname_tag="tsne_raw_perp30_it1000")
plot_tsne(X_raw, y_raw, out_raw, perplexity=50, lr=200, max_iter=500,
          title_text="t-SNE (nenormuota)", fname_tag="tsne_raw_perp50_it1000")
plot_tsne(X_raw, y_raw, out_raw, perplexity=80, lr=200, max_iter=500,
          title_text="t-SNE (nenormuota)", fname_tag="tsne_raw_perp80_it1500")

# 2) Normuota – loop per n_iter ir perplexity tolimensniam loop'e
X_norm, y_norm, out_norm = load_xy(NORM_PATH)
for iters in (250, 300, 350, 500, 600, 750, 1000):
    plot_tsne(X_norm, y_norm, out_norm, perplexity=50, lr=200, max_iter=iters,
              title_text=f"t-SNE (normuota; n_iter={iters})",
              fname_tag=f"tsne_norm_perp50_it{iters}")

for perp in (35, 40):
    plot_tsne(
        X_norm, y_norm, out_norm,
        perplexity=perp, lr=200, max_iter=600,
        title_text=f"t-SNE (normuota; perplexity={perp})",
        fname_tag=f"tsne_norm_perp{perp}_it600"
    )