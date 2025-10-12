# === SPECIFIKOS IR APRAŠOMOSIOS STATISTIKOS ŠABLONAS ===
# Reikia: pandas, numpy, scikit-learn, matplotlib
# Failai:
# - full_clean_with_hw.csv  -> VISOS aibės specifikai (su Height/Weight; visi stulpeliai paversti į skaitinius)
# - clean_data.csv          -> PASIRINKTOS aibės specifikai ir klasėms (be Height/Weight; visi stulpeliai skaitiniai)

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ======= NUSTATYK KELIUS =======
FULL_PATH = "/mnt/data/full_clean_with_hw.csv"   # jei neturit – palik kaip yra; kodas šitą sekciją praleis
SELECTED_PATH = "/mnt/data/clean_data.csv"       # jūsų turimas failas (be Height/Weight)

LABEL_COL = "NObeyesdad"  # klasė 0..6 pagal jūsų aprašymą

# ======= PAGALBINĖS FUNKCIJOS =======

def compute_spec(df: pd.DataFrame, label_col: str, title: str):
    """
    Apskaičiuoja m, n, ρ_n (santykinė vidinė dimensija pagal PCA ≥90% dispersijos),
    γ_n (retumo koef.), ir sukuria PCA paaiškinamos dispersijos kreivę.
    Naudoja Min–Max skalę PCA skaičiavimui, kad būtų suderinta su projekto normalizacija.
    """
    assert label_col in df.columns, f"Label (klasė) stulpelis '{label_col}' nerastas."
    X = df.drop(columns=[label_col]).copy()

    # m, n
    m = X.shape[0]
    n = X.shape[1]

    # γ_n (retumo koef.) = 1 - u/(n*m), kur u = nenulinių reikšmių sk.
    # Pastaba: jei daug 0/1 indikatorių, γ_n bus didesnis (daug nulinių).
    u = np.count_nonzero(X.values)
    gamma_n = 1.0 - (u / (n * m))

    # ρ_n (santykinė vidinė dimensija): kiek PCA komponentų reikia ≥90% dispersijos, dalinta iš n
    scaler = MinMaxScaler()
    X_mm = scaler.fit_transform(X.values)
    pca = PCA(n_components=min(n, m))
    pca.fit(X_mm)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cumsum, 0.90) + 1)  # komponentų sk., reikalingų ≥90%
    rho_n = k / n

    # Spausdinam santrauką
    print(f"=== {title} ===")
    print(f"Tipas: lentelė")
    print(f"m (eilučių sk.): {m}")
    print(f"n (požymių sk.): {n}")
    print(f"Vidinė dimensija: k={k} komponentų (≥90% dispersijos) → ρ_n = {rho_n:.3f}")
    print(f"Retumo koeficientas γ_n = {gamma_n:.3f}  (γ_n∈[0,1], kuo arčiau 1 – tuo retesnė aibė)")
    # Mastelio pastaba
    mins = X.min(numeric_only=True)
    maxs = X.max(numeric_only=True)
    print(f"Mastelis (prieš min–max): medianinis min={mins.median():.3f}, medianinis max={maxs.median():.3f}")
    print()

    # PCA paaiškinamos dispersijos kreivė
    plt.figure()
    plt.plot(np.arange(1, len(cumsum)+1), cumsum, marker='o')
    plt.axhline(0.90, linestyle='--')
    plt.xlabel("PCA komponentų sk.")
    plt.ylabel("Sukaupta paaiškinama dispersija")
    plt.title(f"PCA paaiškinamos dispersijos kreivė – {title} (ρ_n≈{rho_n:.2f})")
    plt.tight_layout()
    out_png = f"./pca_cumsum_{title.replace(' ', '_')}.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[Išsaugota kreivė] {out_png}\n")

def class_table(df: pd.DataFrame, label_col: str, title: str):
    """
    Klasės (NObeyesdad) dažniai ir procentai pasirinktoje aibėje.
    """
    assert label_col in df.columns
    counts = df[label_col].value_counts().sort_index()
    perc = (counts / counts.sum() * 100).round(2)
    tbl = pd.DataFrame({"class": counts.index, "count": counts.values, "percent": perc.values})
    print(f"=== {title}: klasių pasiskirstymas ===")
    print(tbl.to_string(index=False))
    print()
    return tbl

def quick_feature_histograms(df: pd.DataFrame, cols, title_prefix: str):
    """
    Greitos histogramėlės keliems pagrindiniams skaitiniams požymiams (jei egzistuoja).
    """
    for c in cols:
        if c in df.columns:
            plt.figure()
            plt.hist(df[c].dropna().values, bins=20)
            plt.title(f"{title_prefix} – {c}")
            plt.xlabel(c)
            plt.ylabel("Dažnis")
            plt.tight_layout()
            out_png = f"./hist_{title_prefix.replace(' ', '_')}_{c}.png"
            plt.savefig(out_png, dpi=150)
            plt.close()
            print(f"[Išsaugota histograma] {out_png}")

# ======= VYKDYMAS =======

# 3.1 Visos duomenų aibės specifika (jei turit failą su Height/Weight)
if os.path.exists(FULL_PATH):
    df_full = pd.read_csv(FULL_PATH)
    # Tikrinam, ar LABEL_COL egzistuoja ir yra skaitinis 0..6
    if df_full[LABEL_COL].dtype.kind not in "iu":
        print(f"ĮSPĖJIMAS: {LABEL_COL} nėra sveikasis tipas – patikrinkit kodavimą (turi būti 0..6).")
    compute_spec(df_full, LABEL_COL, title="Visos aibės specifika (su Height/Weight)")
else:
    print(f"[Pastaba] Nepavyko rasti {FULL_PATH}. 3.1 (visos aibės specifika) praleidžiama.\n"
          f"Jei norit šią dalį, paruoškit failą su Height/Weight ir visais skaitiniais požymiais.")

# 3.2 Pasirinktos požymių aibės specifika (be Height/Weight)
df_sel = pd.read_csv(SELECTED_PATH)
if df_sel[LABEL_COL].dtype.kind not in "iu":
    print(f"ĮSPĖJIMAS: {LABEL_COL} nėra sveikasis tipas – patikrinkit kodavimą (turi būti 0..6).")
compute_spec(df_sel, LABEL_COL, title="Pasirinktos aibės specifika (be Height/Weight)")

# 3.3 Klasės ir aprašomoji statistika (ant pasirinktos aibės)
tbl = class_table(df_sel, LABEL_COL, title="Pasirinkta aibė")
# Pora bazinių histogramų (jei tokie stulpeliai yra)
quick_feature_histograms(
    df_sel,
    cols=["Age", "NCP", "CH2O", "FCVC", "FAF", "TUE"],  # keiskit pagal tai, kas pas jus yra
    title_prefix="Pasirinkta aibė"
)
print("Done.")
