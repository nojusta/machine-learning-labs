import pandas as pd
import numpy as np

SRC_PATH = './data/clean_data.csv'
CLEAN_NO_OUTLIERS_PATH = './data/clean_data_without_outliers.csv'
NORM_MINMAX_PATH = './data/normalized_minmax.csv'

# Numeric columns to process (present in your cleaned dataset)
CANDIDATE_NUM_COLS = ['Age', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

def find_outlier_indices(df: pd.DataFrame, cols):
    idx = set()
    for col in cols:
        s = pd.to_numeric(df[col], errors='coerce')
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 3 * iqr
        upper = q3 + 3 * iqr
        bad = df[(s < lower) | (s > upper)].index
        idx.update(bad)
    return sorted(idx)

def minmax_normalize(df: pd.DataFrame, cols):
    mins = df[cols].min()
    maxs = df[cols].max()
    scale = (maxs - mins).replace(0, np.nan)  # avoid div by zero
    out = (df[cols].astype(float) - mins) / scale
    return out.fillna(0.0)

# --- Added: descriptive stats helpers ---
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
# --- end added ---

def main():
    df = pd.read_csv(SRC_PATH)

    # Keep only columns that exist
    num_cols = [c for c in CANDIDATE_NUM_COLS if c in df.columns]

    # Ensure numeric dtype for processing (in case CSV stored as strings)
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    print("Eilučių skaičius prieš filtravimą:", len(df))

    # --- Added: stats before outlier removal ---
    print("\nAprašomosios statistikos lentelė (prieš išimčių šalinimą):")
    print(describe(df, num_cols))

    # Outlier removal: drop full rows if any selected numeric col is extreme
    outlier_idx = find_outlier_indices(df, num_cols)
    print("\nIšimčių eilutės (indeksų skaičius):", len(outlier_idx))

    df_no_outliers = df.drop(index=outlier_idx).reset_index(drop=True)
    df_no_outliers.to_csv(CLEAN_NO_OUTLIERS_PATH, index=False)
    print("Išsaugota be išimčių:", CLEAN_NO_OUTLIERS_PATH)
    print("Eilučių skaičius po filtravimo:", len(df_no_outliers))

    # --- Added: stats after outlier removal ---
    print("\nAprašomosios statistikos lentelė (po išimčių šalinimo):")
    print(describe(df_no_outliers, num_cols))

    # Min-Max normalization ONLY on selected numeric columns
    df_norm = df_no_outliers.copy()
    df_norm[num_cols] = minmax_normalize(df_no_outliers, num_cols).round(4)

    df_norm.to_csv(NORM_MINMAX_PATH, index=False)
    print("\nMin-Max normalizuota aibė išsaugota į:", NORM_MINMAX_PATH)

    # Preview
    print("\nPirmos 5 eilutės (tik normalizuoti stulpeliai):")
    print(df_norm[num_cols].head())

if __name__ == '__main__':
    main()