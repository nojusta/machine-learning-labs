import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap

# 1. DETALI LOKALI STRUKTŪRA (išskirčių identifikavimui)
config_1 = {
    'n_neighbors': 15,
    'min_dist': 0.1,
    'spread': 1.0
}

# 2. SUBALANSUOTA (bendram supratimui)
config_2 = {
    'n_neighbors': 25,
    'min_dist': 0.2,
    'spread': 1.0
}

# 3. GLOBALI STRUKTŪRA (kontekstui)
config_3 = {
    'n_neighbors': 40,
    'min_dist': 0.3,
    'spread': 1.5
}

# 4. LABAI GLOBALI STRUKTŪRA (labai bendram supratimui)
config_4 = {
    'n_neighbors': 50,
    'min_dist': 0.5,
    'spread': 1
}

def skaiciuoti_isvestis(df, stulpeliai):
    inner = np.zeros(len(df), dtype=bool)
    outer = np.zeros(len(df), dtype=bool)
    for c in stulpeliai:
        s = pd.to_numeric(df[c], errors='coerce')
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        inner_lo, inner_hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outer_lo, outer_hi = q1 - 3.0 * iqr, q3 + 3.0 * iqr
        mask_inner = (s < inner_lo) | (s > inner_hi)
        mask_outer = (s < outer_lo) | (s > outer_hi)
        inner |= mask_inner.fillna(False).to_numpy()
        outer |= mask_outer.fillna(False).to_numpy()
    return inner, outer

def _minmax_from_clean(clean_df, cols):
    out = clean_df.copy()
    for c in cols:
        s = pd.to_numeric(out[c], errors='coerce')
        mn, mx = s.min(), s.max()
        if pd.isna(mn) or pd.isna(mx) or mx == mn:
            out[c] = 0.0
        else:
            out[c] = (s - mn) / (mx - mn)
    return out

def load_data(pozymiai, naudoti_raw=False):
    if naudoti_raw:
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "data", "clean_data.csv"),
            os.path.join(".", "data", "clean_data.csv"),
            "../data/clean_data.csv"
        ]
    else:
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "data", "normalized_minmax.csv"),
            os.path.join(".", "data", "normalized_minmax.csv"),
            "../data/normalized_minmax.csv"
        ]
    df = None
    for pth in candidates:
        if pth and os.path.isfile(pth):
            try:
                df = pd.read_csv(pth)
                print(f"Užkrautas failas: {pth}")
                break
            except Exception as e:
                print(f"Nepavyko užkrauti {pth}: {e}")
    if df is None and not naudoti_raw:
        # Jei nėra normalizuoto, bandome iš clean_data.csv sugeneruoti min-max
        clean_candidates = [
            os.path.join(os.path.dirname(__file__), "..", "data", "clean_data.csv"),
            os.path.join(".", "data", "clean_data.csv"),
            "../data/clean_data.csv"
        ]
        clean_df = None
        for pth in clean_candidates:
            if os.path.isfile(pth):
                try:
                    clean_df = pd.read_csv(pth)
                    print(f"Užkrautas clean failas: {pth} - sukursiu min-max versiją iš šio.")
                    break
                except Exception as e:
                    print(f"Nepavyko užkrauti {pth}: {e}")
        if clean_df is None:
            raise SystemExit("Nenustatytas normalizuotas failas ir nepavyko rasti clean_data.csv.")
        missing = [c for c in pozymiai if c not in clean_df.columns]
        if missing:
            raise SystemExit(f"Trūksta požymių clean_data.csv: {missing}")
        df = _minmax_from_clean(clean_df, pozymiai)
        if 'NObeyesdad' in clean_df.columns and 'NObeyesdad' not in df.columns:
            df['NObeyesdad'] = clean_df['NObeyesdad']
    missing_cols = [c for c in pozymiai if c not in df.columns]
    if missing_cols:
        raise SystemExit(f"Reikalingi požymiai nerasti faile: {missing_cols}")
    df = df.dropna(subset=pozymiai)
    for c in pozymiai:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def filter_obesity(df):
    allowed = {'4', '5', '6', 4, 5, 6}
    if 'NObeyesdad' in df.columns:
        mask = df['NObeyesdad'].astype(str).isin(allowed)
        return df[mask].reset_index(drop=True)
    return df

def plot_umap(df, pozymiai, n_neighbors, min_dist, sklaida, metric, ax):
    X = df[pozymiai].astype(float).values
    etiketems = df['NObeyesdad'].astype(str).values if 'NObeyesdad' in df.columns else None
    inner_mask, outer_mask = skaiciuoti_isvestis(df, pozymiai)
    umap_modelis = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=sklaida,
        metric=metric,
        random_state=42,
        init='spectral',
        low_memory=True
    )
    embedding = umap_modelis.fit_transform(X)
    unikalios_etiketes = sorted(set(etiketems)) if etiketems is not None else []
    spalvos = plt.cm.get_cmap('tab10', max(3, len(unikalios_etiketes)))
    etikete_i_spalva = {etik: spalvos(i) for i, etik in enumerate(unikalios_etiketes)}
    ax.clear()
    # Pagrindiniai taškai (visi NObeyesdad)
    if etiketems is not None:
        for etik in unikalios_etiketes:
            mask = (etiketems == etik)
            ax.scatter(embedding[mask, 0], embedding[mask, 1], s=30, alpha=0.8, color=etikete_i_spalva[etik], label=etik)
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], s=25, alpha=0.8)
    # Vidinės išskirtys - juoda spalva
    if np.any(inner_mask):
        ax.scatter(embedding[inner_mask, 0], embedding[inner_mask, 1], s=15, color='black', alpha=1.0, label='Vidinės išskirtys', zorder=3)
    # Išorinės išskirtys - raudona spalva
    if np.any(outer_mask):
        ax.scatter(embedding[outer_mask, 0], embedding[outer_mask, 1], s=6, color='red', alpha=1.0, label='Išorinės išskirtys', zorder=4)
    ax.set_title(f'UMAP (NObeyesdad: 4,5,6)\nn_neighbors={n_neighbors}, min_dist={min_dist:.2f}, metric={metric}')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(loc='best', fontsize='small')
    plt.draw()

def main():
    pozymiai = ['Gender', 'FCVC', 'SMOKE', 'CALC', 'NCP', 'CH2O']
    print("Ar naudoti tik raw clean_data.csv (be normalizacijos)? (t/n): ", end="")
    naudoti_raw = input().strip().lower() == 't'
    df = load_data(pozymiai, naudoti_raw=naudoti_raw)
    df = filter_obesity(df)
    print("Pasirinkite konfigūraciją (1 - detali, 2 - subalansuota, 3 - globali, 4 - labai globali): ", end="")
    try:
        config_choice = int(input())
    except Exception:
        config_choice = 2
    if config_choice == 1:
        config = config_1
    elif config_choice == 3:
        config = config_3
    elif config_choice == 4:
        config = config_4
    else:
        config = config_2

    print("Įveskite metriką (euclidean/manhattan/cosine): ", end="")
    metric = input().strip().lower()
    if metric not in ['euclidean', 'manhattan', 'cosine']:
        metric = 'euclidean'

    fig, ax = plt.subplots(figsize=(9, 7))
    plot_umap(df, pozymiai, config['n_neighbors'], config['min_dist'], config['spread'], metric, ax)
    plt.show()

if __name__ == '__main__':
    main()