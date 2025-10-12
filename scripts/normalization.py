import pandas as pd
import numpy as np

SRC_PATH = '../data/clean_data.csv'
NORM_MINMAX_PATH = '../data/normalized_minmax.csv'

CANDIDATE_NUM_COLS = ['Age', 'FCVC', 'NCP', 'CAEC', 'CH2O', 'FAF', 'TUE', 'CALC', 'MTRANS']

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

    num_cols = [c for c in CANDIDATE_NUM_COLS if c in df.columns]
    if not num_cols:
        raise SystemExit("No numeric columns found to normalize. Update CANDIDATE_NUM_COLS if needed.")

    # ensure numeric
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    print("Eilučių skaičius:", len(df))
    print("\nAprašomoji statistika (prieš normalizaciją):")
    print(describe(df, num_cols))

    df_norm = df.copy()
    df_norm[num_cols] = minmax_normalize(df, num_cols).round(4)

    df_norm.to_csv(NORM_MINMAX_PATH, index=False)
    print(f"\nMin-Max normalizuota aibė išsaugota į: {NORM_MINMAX_PATH}")

    print("\nAprašomoji statistika (po normalizacijos):")
    print(describe(df_norm, num_cols))

if __name__ == '__main__':
    main()