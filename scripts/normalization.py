import pandas as pd
import numpy as np

SRC_PATH = '../data/clean_data.csv'
NORM_MINMAX_PATH = '../data/normalized_minmax.csv'

COLUMNS_OUT = ['Gender', 'FCVC', 'SMOKE', 'CALC', 'NCP', 'CH2O', 'NObeyesdad']
LABEL_COL = 'NObeyesdad'
FEATURE_COLS = [c for c in COLUMNS_OUT if c != LABEL_COL]

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
    df = pd.read_csv(SRC_PATH)
    df = df[COLUMNS_OUT].copy()

    for c in FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    print("Eilučių skaičius:", len(df))
    print("\nAprašomoji statistika (prieš normalizaciją) – visi 6 požymiai:")
    print(describe(df, FEATURE_COLS))

    df_norm = df.copy()
    df_norm[FEATURE_COLS] = minmax_normalize(df, FEATURE_COLS).round(4)

    df_norm.to_csv(NORM_MINMAX_PATH, index=False)
    print(f"\nMin–Max normalizuota aibė išsaugota į: {NORM_MINMAX_PATH}")

    print("\nAprašomoji statistika (po normalizacijos) – visi 6 požymiai:")
    print(describe(df_norm, FEATURE_COLS))

if __name__ == '__main__':
    main()
