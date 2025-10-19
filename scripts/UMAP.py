import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    'n_neighbors': 45,
    'min_dist': 0.9,
    'spread': 1
}

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

def load_outliers(df, naudoti_raw=False):
    """Užkrauna išskirtis iš failų vietoj skaičiavimo"""
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "data", "non-norm_outliers.csv" if naudoti_raw else "outliers.csv"),
        os.path.join(".", "data", "non-norm_outliers.csv" if naudoti_raw else "outliers.csv"),
        "../data/non-norm_outliers.csv" if naudoti_raw else "../data/outliers.csv"
    ]
    
    outliers_df = None
    for pth in candidates:
        if pth and os.path.isfile(pth):
            try:
                outliers_df = pd.read_csv(pth)
                print(f"Užkrautas išskirčių failas: {pth}")
                break
            except Exception as e:
                print(f"Nepavyko užkrauti {pth}: {e}")
    
    if outliers_df is None:
        print("DĖMESIO: Išskirčių failas nerastas - išskirtys nebus pažymėtos")
        inner_mask = np.zeros(len(df), dtype=bool)
        outer_mask = np.zeros(len(df), dtype=bool)
        return inner_mask, outer_mask
    
    # Tikriname ar yra outlier_type stulpelis
    if 'outlier_type' not in outliers_df.columns:
        print("DĖMESIO: Išskirčių faile nėra 'outlier_type' stulpelio - išskirtys nebus pažymėtos")
        inner_mask = np.zeros(len(df), dtype=bool)
        outer_mask = np.zeros(len(df), dtype=bool)
        return inner_mask, outer_mask
    
    if len(outliers_df) != len(df):
        print(f"DĖMESIO: Išskirčių faile yra {len(outliers_df)} eilutės, o duomenų faile {len(df)} eilutės")
        print("Išskirtys bus pritaikytos pirmosioms bendrosioms eilutėms")
    
    # Imame pirmuosius len(df) įrašus arba visus, jei jų mažiau
    n = min(len(df), len(outliers_df))
    inner_mask = np.zeros(len(df), dtype=bool)
    outer_mask = np.zeros(len(df), dtype=bool)
    
    # outlier_type: 0 - ne isskirtis, 1 - vidine isskirtis, 2 - issorine isskirtis
    outlier_values = outliers_df['outlier_type'].values[:n]
    inner_mask[:n] = (outlier_values == 1)
    outer_mask[:n] = (outlier_values == 2)
    
    return inner_mask, outer_mask
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

def plot_umap(df, pozymiai, n_neighbors, min_dist, sklaida, metric, ax, naudoti_raw=False):
    X = df[pozymiai].astype(float).values
    etiketems = df['NObeyesdad'].astype(str).values if 'NObeyesdad' in df.columns else None
    
    # Naudojame užkrautas išskirtis vietoj skaičiavimo
    inner_mask, outer_mask = load_outliers(df, naudoti_raw)
    
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
    
    # Fiksuotos spalvos klasėms
    class_colors = {
        '4': "#ff7f0e",
        '5': "#aec7e8",
        '6': "#2ca02c",
        4: "#ff7f0e",
        5: "#aec7e8",
        6: "#2ca02c"
    }
    
    ax.clear()
    # Pagrindiniai taškai (visi NObeyesdad)
    if etiketems is not None:
        unikalios_etiketes = sorted(set(etiketems))
        for etik in unikalios_etiketes:
            mask = (etiketems == etik)
            color = class_colors.get(etik, class_colors.get(str(etik), '#1f77b4'))  # default mėlyna
            ax.scatter(embedding[mask, 0], embedding[mask, 1], s=30, alpha=0.8, 
                      color=color, label=f"Klasė {etik}")
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], s=25, alpha=0.8)
        
    # Vidinės išskirtys - juoda TUŠČIAVIDURĖ forma (apskritimas)
    if np.any(inner_mask):
        ax.scatter(embedding[inner_mask, 0], embedding[inner_mask, 1], 
                  s=70, facecolors='none', edgecolors='black', 
                  linewidths=1.5, label='Vidinės išskirtys', zorder=3)
                  
    # Išorinės išskirtys - raudona TUŠČIAVIDURĖ forma (apskritimas)
    if np.any(outer_mask):
        ax.scatter(embedding[outer_mask, 0], embedding[outer_mask, 1], 
                  s=70, facecolors='none', edgecolors='red',
                  linewidths=1.5, label='Išorinės išskirtys', zorder=4)
                  
    data_type = "raw" if naudoti_raw else "min-max"
    ax.set_title(f'UMAP ({data_type}, NObeyesdad: 4,5,6)\nn_neighbors={n_neighbors}, min_dist={min_dist:.2f}, metric={metric}')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(loc='best', fontsize='small')
    plt.draw()

def main():
    pozymiai = ['Gender', 'FCVC', 'SMOKE', 'CALC', 'NCP', 'CH2O']
    print("Ar naudoti tik raw clean_data.csv (be normalizacijos)? (t/n): ", end="")
    naudoti_raw = input().strip().lower() == 't'
    
    # 1. Užkrauname pilnus duomenis
    df = load_data(pozymiai, naudoti_raw=naudoti_raw)
    
    # 2. Užkrauname išskirtis PRIEŠ filtravimą
    full_outliers_df = None
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "data", "non-norm_outliers.csv" if naudoti_raw else "outliers.csv"),
        os.path.join(".", "data", "non-norm_outliers.csv" if naudoti_raw else "outliers.csv"),
        "../data/non-norm_outliers.csv" if naudoti_raw else "../data/outliers.csv"
    ]
    
    for pth in candidates:
        if pth and os.path.isfile(pth):
            try:
                full_outliers_df = pd.read_csv(pth)
                print(f"Užkrautas išskirčių failas: {pth}")
                break
            except Exception as e:
                print(f"Nepavyko užkrauti {pth}: {e}")
    
    if full_outliers_df is None:
        print("DĖMESIO: Išskirčių failas nerastas")
    elif 'outlier_type' not in full_outliers_df.columns:
        print("DĖMESIO: Išskirčių faile nėra 'outlier_type' stulpelio")
    elif len(full_outliers_df) != len(df):
        print(f"DĖMESIO: Išskirčių faile yra {len(full_outliers_df)} eilutės, o pilname duomenų faile {len(df)} eilutės")
    
    # 3. Filtruojame duomenis IR išskirtis pagal NObeyesdad
    allowed = {'4', '5', '6', 4, 5, 6}
    if 'NObeyesdad' in df.columns:
        mask = df['NObeyesdad'].astype(str).isin(allowed)
        df = df[mask].reset_index(drop=True)
        
        # Filtruojame ir išskirtis, jei jų turime
        if full_outliers_df is not None:
            full_outliers_df = full_outliers_df[mask].reset_index(drop=True)
            print(f"Duomenys ir išskirtys filtruoti pagal NObeyesdad 4,5,6: {len(df)} eilutės")
    
    # Sukuriame išskirčių maskes iš filtruotų duomenų
    inner_mask = np.zeros(len(df), dtype=bool)
    outer_mask = np.zeros(len(df), dtype=bool)
    if full_outliers_df is not None and 'outlier_type' in full_outliers_df.columns:
        outlier_values = full_outliers_df['outlier_type'].values
        inner_mask = (outlier_values == 1)
        outer_mask = (outlier_values == 2)
    
    # 4. Konfigūracijos pasirinkimas
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

    # 5. Vizualizacija su jau filtruotais duomenimis ir išskirtimis
    fig, ax = plt.subplots(figsize=(9, 7))
    plot_umap_filtered(df, pozymiai, config['n_neighbors'], config['min_dist'], config['spread'], 
                      metric, ax, naudoti_raw, inner_mask, outer_mask)
    plt.show()

from matplotlib.lines import Line2D

def plot_umap_filtered(df, pozymiai, n_neighbors, min_dist, sklaida, metric, ax, naudoti_raw=False, inner_mask=None, outer_mask=None):
    X = df[pozymiai].astype(float).values
    etiketems = df['NObeyesdad'].astype(str).values if 'NObeyesdad' in df.columns else None
    
    # Naudojame jau filtruotas išskirčių maskes
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
    
    # Fiksuotos spalvos klasėms
    class_colors = {
        '4': "#F399FF",  # rožinė (tipo 1)
        '5': "#0080DB",  # mėlyna (tipo 2)
        '6': "#48A348",  # žalia (tipo 3)
        4: "#F399FF",
        5: "#0080DB",
        6: "#48A348"
    }
    
    ax.clear()
    # Pagrindiniai taškai (visi NObeyesdad)
    if etiketems is not None:
        unikalios_etiketes = sorted(set(etiketems))
        
        # Pirma vaizduojame ne-išskirtis
        for etik in unikalios_etiketes:
            # Taškai, kurie nėra nei inner, nei outer išskirtys
            mask = (etiketems == etik) & (~inner_mask) & (~outer_mask)
            color = class_colors.get(etik, class_colors.get(str(etik), '#1f77b4'))
            ax.scatter(embedding[mask, 0], embedding[mask, 1], s=30, alpha=0.8, 
                      color=color)
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], s=25, alpha=0.8)
    
    # Vidinės išskirtys - su juodu apvadu
    if np.any(inner_mask):
        for etik in unikalios_etiketes:
            mask = (etiketems == etik) & inner_mask
            if np.any(mask):
                color = class_colors.get(etik, class_colors.get(str(etik), '#1f77b4'))
                ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                          s=55, facecolor=color, edgecolor='black', 
                          linewidth=1.1, alpha=0.95, zorder=3)
    
    # Išorinės išskirtys - su raudonu apvadu
    if np.any(outer_mask):
        for etik in unikalios_etiketes:
            mask = (etiketems == etik) & outer_mask
            if np.any(mask):
                color = class_colors.get(etik, class_colors.get(str(etik), '#1f77b4'))
                ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                          s=55, facecolor=color, edgecolor='red', 
                          linewidth=1.3, alpha=0.95, zorder=4)
    
    # Legenda su kontūrais
    legend_elements = []
    for etik in sorted(unikalios_etiketes):
        color = class_colors.get(etik, class_colors.get(str(etik), '#1f77b4'))
        legend_elements.append(
            Line2D([0], [0], marker='o', linestyle='None',
                   markerfacecolor=color, markeredgecolor='none',
                   label=f'Nutukimo tipas {etik}', markersize=9)
        )
    
    # Pridedame išskirčių legendas
    legend_elements.append(
        Line2D([0], [0], marker='o', linestyle='None',
               markerfacecolor='white', markeredgecolor='black',
               label='Vidinė išskirtis', markersize=9, markeredgewidth=1.3)
    )
    legend_elements.append(
        Line2D([0], [0], marker='o', linestyle='None',
               markerfacecolor='white', markeredgecolor='red',
               label='Išorinė išskirtis', markersize=9, markeredgewidth=1.3)
    )
    
    data_type = "raw" if naudoti_raw else "min-max"
    ax.set_title(f'UMAP ({data_type}, NObeyesdad: 4,5,6)\nn_neighbors={n_neighbors}, min_dist={min_dist:.2f}, metric={metric}')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(handles=legend_elements, title="Kategorijos", frameon=True, loc='best', fontsize='small')
    plt.draw()

if __name__ == '__main__':
    main()