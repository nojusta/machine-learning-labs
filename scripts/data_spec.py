import os
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

FULL_PATH = "../data/full_clean_with_hw.csv"
SELECTED_PATH = "../data/clean_data.csv"
LABEL_COL = "NObeyesdad"
OUT_DIR = "../outputs/spec"
os.makedirs(OUT_DIR, exist_ok=True)

def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)

def spec_report(df: pd.DataFrame, label_col: str, title: str):
    if label_col not in df.columns:
        raise ValueError(f"Trūksta stulpelio: {label_col}")
    X = df.drop(columns=[label_col])

    m, n = X.shape
    nnz = np.count_nonzero(X.to_numpy())
    gamma_n = 1.0 - nnz / (n * m)

    X_mm = MinMaxScaler().fit_transform(X)
    pca = PCA(n_components=min(n, m)).fit(X_mm)
    csum = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(csum, 0.90) + 1)
    rho_n = k / n

    mins = X.min(numeric_only=True)
    maxs = X.max(numeric_only=True)

    print(f"=== {title} ===")
    print(f"Tipas: lentelė")
    print(f"m: {m}")
    print(f"n: {n}")
    print(f"k (≥90%): {k}  →  ρ_n = {rho_n:.3f}")
    print(f"γ_n = {gamma_n:.3f}")
    print(f"Medianinis min: {mins.median():.3f} | medianinis max: {maxs.median():.3f}\n")

    out_png = os.path.join(OUT_DIR, f"pca_cumsum_{slug(title)}.png")
    plt.figure()
    plt.plot(np.arange(1, len(csum) + 1), csum, marker='o')
    plt.axhline(0.90, linestyle='--')
    plt.xlabel("PCA komponentų sk.")
    plt.ylabel("Sukaupta paaiškinama dispersija")
    plt.title(f"PCA kreivė – {title} (ρ_n≈{rho_n:.2f})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Išsaugota: {out_png}\n")

def class_counts(df: pd.DataFrame, label_col: str, title: str):
    if label_col not in df.columns:
        raise ValueError(f"Trūksta stulpelio: {label_col}")
    counts = df[label_col].value_counts().sort_index()
    perc = (counts / counts.sum() * 100).round(2)
    tbl = pd.DataFrame({"class": counts.index, "count": counts.values, "percent": perc.values})
    print(f"=== {title}: klasių pasiskirstymas ===")
    print(tbl.to_string(index=False))
    print()
    return tbl

def save_hists(df: pd.DataFrame, cols, title_prefix: str):
    pref = slug(title_prefix)
    for c in cols:
        if c not in df.columns:
            continue
        out_png = os.path.join(OUT_DIR, f"hist_{pref}_{slug(c)}.png")
        plt.figure()
        plt.hist(df[c].dropna().to_numpy(), bins=20)
        plt.title(f"{title_prefix} – {c}")
        plt.xlabel(c)
        plt.ylabel("Dažnis")
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"Išsaugota: {out_png}")

if os.path.exists(FULL_PATH):
    df_full = pd.read_csv(FULL_PATH)
    if df_full[LABEL_COL].dtype.kind not in "iu":
        print(f"Įspėjimas: {LABEL_COL} nėra sveikasis tipas (turėtų būti 0..6).")
    spec_report(df_full, LABEL_COL, "Visos aibės specifika")
    class_counts(df_full, LABEL_COL, "Visos aibės")
    class_counts(df_full[df_full[LABEL_COL].isin([4, 5, 6])], LABEL_COL, "Visos aibės (tik klasės 4–6)")
else:
    print(f"Pastaba: {FULL_PATH} nerastas. 3.1 praleidžiama.")

df_sel = pd.read_csv(SELECTED_PATH)
if df_sel[LABEL_COL].dtype.kind not in "iu":
    print(f"Įspėjimas: {LABEL_COL} nėra sveikasis tipas (turėtų būti 0..6).")
spec_report(df_sel, LABEL_COL, "Pasirinktos aibės specifika")

class_counts(df_sel, LABEL_COL, "Pasirinkta aibė")
class_counts(df_sel[df_sel[LABEL_COL].isin([4, 5, 6])], LABEL_COL, "Pasirinkta aibė (tik klasės 4–6)")
