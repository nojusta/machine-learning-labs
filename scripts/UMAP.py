import argparse
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser(description="UMAP only visualization (raw vs min-max).")
    p.add_argument("--n_neighbors", type=int, default=50)
    p.add_argument("--min_dist", type=float, default=1)
    p.add_argument("--spread", type=float, default=1.0)
    p.add_argument("--random_state", type=lambda x: None if x.lower()=="none" else int(x), default=42)
    p.add_argument("--raw", type=str, default="../data/clean_data.csv", help="raw (unnormalized) input")
    p.add_argument("--norm", type=str, default="../data/normalized_minmax.csv", help="min-max normalized input (if missing, will be computed)")
    p.add_argument("--subset", type=str, default="behavioral", choices=["all","behavioral","demographic","custom"], help="feature subset")
    p.add_argument("--custom_cols", type=str, default="", help="comma-separated if subset=custom")
    p.add_argument("--output", type=str, default=None, help="save figure to PNG (optional)")
    return p.parse_args()

def get_subset_columns(df, name, custom_csv):
    behavioral = ['FCVC','NCP','CH2O','FAF','TUE','CAEC','CALC','SMOKE','SCC','family_history_with_overweight','FAVC','MTRANS']
    demographic = ['Gender','Age']
    if name == "all":
        cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != 'NObeyesdad']
    elif name == "behavioral":
        cols = [c for c in behavioral if c in df.columns]
    elif name == "demographic":
        cols = [c for c in demographic if c in df.columns]
    else:  # custom
        custom = [c.strip() for c in custom_csv.split(",") if c.strip()]
        cols = [c for c in custom if c in df.columns]
    # ensure numeric
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_integer_dtype(df[c])]
    return cols

def compute_outliers(df, cols):
    # returns two boolean masks: inner_outlier (1.5*IQR), outer_outlier (3*IQR)
    inner = np.zeros(len(df), dtype=bool)
    outer = np.zeros(len(df), dtype=bool)
    for c in cols:
        s = pd.to_numeric(df[c], errors='coerce')
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        inner_lo, inner_hi = q1 - 1.5*iqr, q3 + 1.5*iqr
        outer_lo, outer_hi = q1 - 3.0*iqr, q3 + 3.0*iqr
        mask_inner = (s < inner_lo) | (s > inner_hi)
        mask_outer = (s < outer_lo) | (s > outer_hi)
        inner = inner | mask_inner.fillna(False).to_numpy()
        outer = outer | mask_outer.fillna(False).to_numpy()
    # outer is subset of inner; keep flags separate
    return inner, outer

def ensure_numeric(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

def minmax_on_the_fly(df, cols):
    out = df.copy()
    for c in cols:
        s = pd.to_numeric(out[c], errors='coerce')
        mn, mx = s.min(), s.max()
        if pd.isna(mn) or pd.isna(mx) or mx==mn:
            out[c] = 0.0
        else:
            out[c] = (s - mn) / (mx - mn)
    return out

def make_umap_embedding(X, n_neighbors, min_dist, spread, random_state):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, spread=spread, random_state=random_state)
    return reducer.fit_transform(X)

def plot_umap(ax, embedding, labels, inner_mask, outer_mask, title):
    # base points colored by label
    if labels is not None:
        sc = ax.scatter(embedding[:,0], embedding[:,1], c=labels, cmap='tab10', s=18, alpha=0.8)
    else:
        sc = ax.scatter(embedding[:,0], embedding[:,1], s=18, alpha=0.8)
    # mark inner outliers (1.5*IQR) with black edge
    if inner_mask.any():
        ax.scatter(embedding[inner_mask,0], embedding[inner_mask,1], facecolors='none', edgecolors='black', s=60, linewidths=0.8, label='inner_outlier')
    # mark outer outliers (3*IQR) with red cross
    if outer_mask.any():
        ax.scatter(embedding[outer_mask,0], embedding[outer_mask,1], marker='x', c='red', s=50, linewidths=1.0, label='outer_outlier')
    ax.set_title(title)
    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
    # legend for outliers only
    handles, labels_ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='best', fontsize='small')

def main():
    args = parse_args()

    # validate min_dist/spread
    args.min_dist = max(0.0, min(1.0, float(args.min_dist)))
    args.spread = float(args.spread)
    if args.spread < args.min_dist:
        args.spread = args.min_dist

    # load raw
    df_raw = pd.read_csv(args.raw)
    # try load normalized; if missing compute on the fly
    try:
        df_norm = pd.read_csv(args.norm)
    except Exception:
        df_norm = None

    # choose columns based on raw (assume same schema)
    cols = get_subset_columns(df_raw, args.subset, args.custom_cols)
    if not cols:
        raise SystemExit("No numeric columns selected for UMAP. Choose different subset/custom columns.")

    # ensure numeric
    ensure_numeric(df_raw, cols)
    if df_norm is None:
        df_norm = minmax_on_the_fly(df_raw, cols)
    else:
        ensure_numeric(df_norm, cols)

    # labels for coloring
    label_col = 'NObeyesdad' if 'NObeyesdad' in df_raw.columns else None
    labels_raw = df_raw[label_col].values if label_col else None
    labels_norm = df_norm[label_col].values if label_col and label_col in df_norm.columns else labels_raw

    # compute outliers on raw (use raw numeric values)
    inner_raw, outer_raw = compute_outliers(df_raw, cols)
    inner_norm, outer_norm = compute_outliers(df_norm, cols)

    # compute embeddings
    X_raw = df_raw[cols].astype(float).values
    X_norm = df_norm[cols].astype(float).values

    emb_raw = make_umap_embedding(X_raw, args.n_neighbors, args.min_dist, args.spread, args.random_state)
    emb_norm = make_umap_embedding(X_norm, args.n_neighbors, args.min_dist, args.spread, args.random_state)

    # plot side-by-side
    fig, axes = plt.subplots(1,2, figsize=(14,6))
    plot_umap(axes[0], emb_raw, labels_raw, inner_raw, outer_raw, f"UMAP (raw) — subset={args.subset}")
    plot_umap(axes[1], emb_norm, labels_norm, inner_norm, outer_norm, f"UMAP (min-max) — subset={args.subset}")

    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Saved {args.output}")
    plt.show()

if __name__ == "__main__":
    main()