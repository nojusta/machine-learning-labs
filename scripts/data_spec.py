import os
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ======= NUSTATYK KELIUS =======
FULL_PATH = "../data/full_clean_with_hw.csv"
SELECTED_PATH = "../data/clean_data.csv"
LABEL_COL = "NObeyesdad"

# Kur dėti paveikslus
OUTPUT_DIR = "../outputs/spec"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _safe_name(s: str) -> str:
    # pakeičia viską, kas ne raidė/skaičius/taškas/pabraukimas/minusas, į "_"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)

def compute_spec(df: pd.DataFrame, label_col: str, title: str):
    assert label_col in df.columns, f"Label (klasė) stulpelis '{label_col}' nerastas."
    X = df.drop(columns=[label_col]).copy()

    m, n = X.shape
    u = np.count_nonzero(X.values)
    gamma_n = 1.0 - (u / (n * m))

    scaler = MinMaxScaler()
    X_mm = scaler.fit_transform(X.values)
    pca = PCA(n_components=min(n, m))
    pca.fit(X_mm)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cumsum, 0.90) + 1)
    rho_n = k / n

    print(f"=== {title} ===")
    print(f"Tipas: lentelė")
    print(f"m (eilučių sk.): {m}")
    print(f"n (požymių sk.): {n}")
    print(f"Vidinė dimensija: k={k} komponentų (≥90% dispersijos) → ρ_n = {rho_n:.3f}")
    print(f"Retumo koeficientas γ_n = {gamma_n:.3f}  (γ_n∈[0,1], kuo arčiau 1 – tuo retesnė aibė)")
    mins = X.min(numeric_only=True)
    maxs = X.max(numeric_only=True)
    print(f"Mastelis (prieš min–max): medianinis min={mins.median():.3f}, medianinis max={maxs.median():.3f}\n")

    # Saugus failo vardas + saugi vieta
    safe_title = _safe_name(title)
    out_png = os.path.join(OUTPUT_DIR, f"pca_cumsum_{safe_title}.png")

    plt.figure()
    plt.plot(np.arange(1, len(cumsum)+1), cumsum, marker='o')
    plt.axhline(0.90, linestyle='--')
    plt.xlabel("PCA komponentų sk.")
    plt.ylabel("Sukaupta paaiškinama dispersija")
    plt.title(f"PCA paaiškinamos dispersijos kreivė – {title} (ρ_n≈{rho_n:.2f})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[Išsaugota kreivė] {out_png}\n")

def class_table(df: pd.DataFrame, label_col: str, title: str):
    assert label_col in df.columns
    counts = df[label_col].value_counts().sort_index()
    perc = (counts / counts.sum() * 100).round(2)
    tbl = pd.DataFrame({"class": counts.index, "count": counts.values, "percent": perc.values})
    print(f"=== {title}: klasių pasiskirstymas ===")
    print(tbl.to_string(index=False))
    print()
    return tbl

def quick_feature_histograms(df: pd.DataFrame, cols, title_prefix: str):
    safe_prefix = _safe_name(title_prefix)
    for c in cols:
        if c in df.columns:
            out_png = os.path.join(OUTPUT_DIR, f"hist_{safe_prefix}_{_safe_name(c)}.png")
            plt.figure()
            plt.hist(df[c].dropna().values, bins=20)
            plt.title(f"{title_prefix} – {c}")
            plt.xlabel(c)
            plt.ylabel("Dažnis")
            plt.tight_layout()
            plt.savefig(out_png, dpi=150)
            plt.close()
            print(f"[Išsaugota histograma] {out_png}")

# ======= VYKDYMAS =======
if os.path.exists(FULL_PATH):
    df_full = pd.read_csv(FULL_PATH)
    if df_full[LABEL_COL].dtype.kind not in "iu":
        print(f"ĮSPĖJIMAS: {LABEL_COL} nėra sveikasis tipas – patikrinkit kodavimą (turi būti 0..6).")
    compute_spec(df_full, LABEL_COL, title="Visos aibės specifika (su Height-Weight)")
else:
    print(f"[Pastaba] Nepavyko rasti {FULL_PATH}. 3.1 praleidžiama.")

df_sel = pd.read_csv(SELECTED_PATH)
if df_sel[LABEL_COL].dtype.kind not in "iu":
    print(f"ĮSPĖJIMAS: {LABEL_COL} nėra sveikasis tipas – patikrinkit kodavimą (turi būti 0..6).")
compute_spec(df_sel, LABEL_COL, title="Pasirinktos aibės specifika (be Height-Weight)")

tbl = class_table(df_sel, LABEL_COL, title="Pasirinkta aibė")
quick_feature_histograms(
    df_sel,
    cols=["Age", "NCP", "CH2O", "FCVC", "FAF", "TUE"],
    title_prefix="Pasirinkta aibė"
)
print("Done.")
