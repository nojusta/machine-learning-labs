import pandas as pd

IN_PATH = "./data/data.csv"
OUT_PATH = "./data/clean_data.csv"
ANOMALY_REPORT_PATH = "./data/anomalies_report.csv" # nelogisku duomenu ataskaita
WINSORIZE = True 

def map_values(series: pd.Series, mapping: dict):
    return (
        series.astype(str)
              .str.strip()
              .str.lower()
              .map(mapping)
              .astype("Int64")
    )

df = pd.read_csv(IN_PATH, encoding="utf-8")

binary_cols = [
    "family_history_with_overweight", "FAVC", "SMOKE", "SCC"
]
numeric_cols = ["Age", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
freq_ordinal_cols = ["CAEC", "CALC"] 
mtrans_col = "MTRANS"
gender_col = "Gender"
target_col = "NObeyesdad"

# Mappings
yn_map = {"no": 0, "yes": 1}
gender_map = {"male": 0, "female": 1}
freq_map = {"no": 0, "sometimes": 1, "frequently": 2, "always": 3}
mtrans_map = {
    "walking": 0,
    "bike": 1,
    "motorbike": 2,
    "automobile": 3,
    "public_transportation": 4,
}

# (light -> heavy)
target_map = {
    "insufficient_weight": 0,
    "normal_weight": 1,
    "overweight_level_i": 2,
    "overweight_level_ii": 3,
    "obesity_type_i": 4,
    "obesity_type_ii": 5,
    "obesity_type_iii": 6,
}
# --- Data Cleaning Steps ---
for c in binary_cols:
    if c in df.columns:
        df[c] = map_values(df[c], yn_map)

if gender_col in df.columns:
    df[gender_col] = map_values(df[gender_col], gender_map)

for c in freq_ordinal_cols:
    if c in df.columns:
        df[c] = map_values(df[c], freq_map)

if mtrans_col in df.columns:
    df[mtrans_col] = map_values(df[mtrans_col], mtrans_map)

for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Logikos patikra
ranges = {
    "Age": (14, 80),
    "Height": (1.30, 2.10),
    "Weight": (30, 250),
    "FCVC": (1, 3),
    "NCP": (1, 4),
    "CH2O": (1, 3),
    "FAF": (0, 3),
    "TUE": (0, 2),
}
ordinal_ranges = {c: (0, 3) for c in freq_ordinal_cols}

anomaly_records = []

def clip_and_record(frame: pd.DataFrame, col: str, lo: float, hi: float):
    s = pd.to_numeric(frame[col], errors="coerce")
    mask_low = s < lo
    mask_high = s > hi
    if mask_low.any() or mask_high.any():
        # record anomalies
        for idx, val in s[mask_low | mask_high].items():
            clipped_to = lo if pd.notna(val) and val < lo else hi
            anomaly_records.append({
                "row_index": idx,
                "column": col,
                "original_value": val,
                "clipped_to": clipped_to if WINSORIZE else None,
                "lower_bound": lo,
                "upper_bound": hi,
            })
        if WINSORIZE:
            frame.loc[mask_low, col] = lo
            frame.loc[mask_high, col] = hi

for col, (lo, hi) in ranges.items():
    if col in df.columns:
        clip_and_record(df, col, lo, hi)

for col, (lo, hi) in ordinal_ranges.items():
    if col in df.columns:
        clip_and_record(df, col, lo, hi)

df = df.drop(columns=["Height", "Weight", "BMI"], errors="ignore")

# Age as integer (Int64)
if "Age" in df.columns:
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce").round().astype("Int64")


# Ensure numeric types for remaining numeric_cols
for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64" if c in ["FCVC","NCP","CH2O","FAF","TUE"] else "Int64")


if target_col in df.columns:
    df[target_col] = (
        df[target_col].astype(str).str.strip().str.lower().map(target_map).astype("Int64")
    )

if anomaly_records:
    pd.DataFrame(anomaly_records).to_csv(ANOMALY_REPORT_PATH, index=False)

def _fmt_float_str(x):
    if pd.isna(x):
        return ""
    s = f"{x:.2f}"
    return s.rstrip("0").rstrip(".")

df_out = df.copy()
float_cols = df_out.select_dtypes(include="float").columns
if len(float_cols) > 0:
    df_out[float_cols] = df_out[float_cols].applymap(_fmt_float_str)

df_out.to_csv(OUT_PATH, index=False, na_rep="")

# Spausdinimas
print(f"Clean data saved to {OUT_PATH}")
if anomaly_records:
    by_col = pd.DataFrame(anomaly_records).groupby("column").size().to_dict()
    print("Anomalies found (per column):", by_col)
    print(f"Details saved to {ANOMALY_REPORT_PATH}")
else:
    print("No anomalies found outside specified ranges.")

# Aprašomoji statistika
numeric_for_stats = df.select_dtypes(include="number").columns.tolist()
if target_col in numeric_for_stats:
    numeric_for_stats.remove(target_col)

stats = df[numeric_for_stats].agg(['min', 'max', 'mean', 'median', 'var'])
stats.loc['1 kvartilė'] = df[numeric_for_stats].quantile(0.25)
stats.loc['3 kvartilė'] = df[numeric_for_stats].quantile(0.75)
stats = stats.rename(index={
    'min': 'Min',
    'max': 'Max',
    'mean': 'Vidurkis',
    'median': 'Mediana',
    'var': 'Dispersija'
})

def _fmt_stat(x):
    if pd.isna(x):
        return "NaN"
    # show integers without .00, others with 2 decimals
    try:
        fx = float(x)
        return f"{int(fx)}" if fx.is_integer() else f"{fx:.2f}"
    except Exception:
        return str(x)

for col in stats.columns:
    stats[col] = stats[col].apply(_fmt_stat)

print("\nAprašomosios statistikos lentelė:")
print(stats)

print("\nPirmos eilutės:")
print(df.head())
print("\nTipai:")
print(df.dtypes)

if anomaly_records:
    print("Anomalies found (per column):", by_col)
    print(f"Details saved to {ANOMALY_REPORT_PATH}")
else:
    print("No anomalies found outside specified ranges.")
print(df.head())
print(df.dtypes)
