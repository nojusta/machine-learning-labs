import pandas as pd
import numpy as np

SRC_PATH = './data/clean_data_full.csv'
NORM_MINMAX_PATH = './data/normalized_minmax_all.csv'

# Visi stulpeliai, išskyrus target kintamąjį
LABEL_COL = 'NObeyesdad'

def minmax_normalize(df: pd.DataFrame, cols):
    mins = df[cols].min()
    maxs = df[cols].max()
    scale = (maxs - mins).replace(0, np.nan)
    out = (df[cols].astype(float) - mins) / scale
    return out.fillna(0.0)

def _fmt_number(x):
    if pd.isna(x):
        return "NaN"
    fx = float(x)
    return f"{int(fx)}" if fx.is_integer() else f"{fx:.2f}"

def describe(df: pd.DataFrame, cols):
    stats = df[cols].agg(['min', 'max', 'mean', 'median', 'var'])
    stats.loc['1 kvartilė'] = df[cols].quantile(0.25)
    stats.loc['3 kvartilė'] = df[cols].quantile(0.75)
    stats = stats.rename(index={
        'min': 'Min',
        'max': 'Max',
        'mean': 'Vidurkis',
        'median': 'Mediana',
        'var': 'Dispersija'
    })
    for c in cols:
        stats[c] = stats[c].apply(_fmt_number)
    return stats

def main():
    print("Normalizavimo procesas pradėtas...")
    
    df = pd.read_csv(SRC_PATH)
    print(f"Užkrauta {len(df)} eilučių, {len(df.columns)} stulpelių")
    
    # Identifikuojame visus stulpelius
    all_columns = list(df.columns)
    print(f"Visi stulpeliai: {all_columns}")
    
    # Pašaliname target kintamąjį iš normalizavimo
    if LABEL_COL in all_columns:
        feature_cols = [c for c in all_columns if c != LABEL_COL]
        columns_out = all_columns  # Išlaikome visus stulpelius
        print(f"Target kintamasis: {LABEL_COL}")
    else:
        feature_cols = all_columns
        columns_out = all_columns
        print("Target kintamasis nerastas - normalizuojami visi stulpeliai")
    
    print(f"Normalizuojami požymiai ({len(feature_cols)}): {feature_cols}")
    
    # Pasiimame tik reikalingus stulpelius
    df = df[columns_out].copy()
    
    # Konvertuojame požymius į skaitinius
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        print(f"Konvertuotas į skaitinį: {c}")
    
    print(f"\nEilučių skaičius: {len(df)}")
    print(f"\nAprašomoji statistika (prieš normalizaciją) – visi {len(feature_cols)} požymiai:")
    
    # Parodome statistiką tik pirmų 10 stulpelių (kad neužgriūtų)
    display_cols = feature_cols[:10] if len(feature_cols) > 10 else feature_cols
    print(describe(df, display_cols))
    
    if len(feature_cols) > 10:
        print(f"... ir dar {len(feature_cols) - 10} stulpelių")
    
    # Normalizuojame
    print("\nVykdoma Min-Max normalizacija...")
    df_norm = df.copy()
    df_norm[feature_cols] = minmax_normalize(df, feature_cols).round(4)
    
    # Išsaugome
    df_norm.to_csv(NORM_MINMAX_PATH, index=False)
    print(f"\nMin–Max normalizuota aibė išsaugota į: {NORM_MINMAX_PATH}")
    
    print(f"\nAprašomoji statistika (po normalizacijos) – visi {len(feature_cols)} požymiai:")
    print(describe(df_norm, display_cols))
    
    if len(feature_cols) > 10:
        print(f"... ir dar {len(feature_cols) - 10} stulpelių")
    
    # Patikrinimas - ar normalizacija pavyko
    norm_stats = df_norm[feature_cols].agg(['min', 'max'])
    all_mins_zero = (norm_stats.loc['min'] == 0).all()
    all_maxs_one = (norm_stats.loc['max'] == 1).all()
    
    print(f"\nNormalizacijos patikrinimas:")
    print(f"  Visi minimumi = 0: {'Taip' if all_mins_zero else 'Ne'}")
    print(f"  Visi maksimumai = 1: {'Taip' if all_maxs_one else 'Ne'}")
    
    # Trūkstamų reikšmių patikrinimas
    missing_before = df[feature_cols].isnull().sum().sum()
    missing_after = df_norm[feature_cols].isnull().sum().sum()
    print(f"  Trūkstamos reikšmės (prieš): {missing_before}")
    print(f"  Trūkstamos reikšmės (po): {missing_after}")
    
    print(f"\nGalutinė duomenų forma: {df_norm.shape}")
    print("Normalizavimo procesas baigtas!")
    
    return df_norm

if __name__ == '__main__':
    normalized_data = main()