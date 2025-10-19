import pandas as pd
import numpy as np

IN_PATH = "../data/data.csv"
OUT_PATH = "../data/clean_data.csv"
ANOMALY_REPORT_PATH = "../data/anomalies_report.csv"
WINSORIZE = True

def _fmt_float_str(x):
    if pd.isna(x):
        return ""
    try:
        fx = float(x)
    except Exception:
        return str(x)
    s = f"{fx:.2f}"
    return s.rstrip("0").rstrip(".")

def map_values(series: pd.Series, mapping: dict):
    return (
        series.astype(str)
              .str.strip()
              .str.lower()
              .map(mapping)
              .astype("Int64")
    )

def clip_and_record(frame: pd.DataFrame, col: str, lo: float, hi: float, report: list):
    s = pd.to_numeric(frame[col], errors="coerce")
    mask_low = s < lo
    mask_high = s > hi
    if mask_low.any() or mask_high.any():
        for idx, val in s[mask_low | mask_high].items():
            clipped_to = lo if (pd.notna(val) and val < lo) else hi
            report.append({
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

binary_cols = ["family_history_with_overweight", "FAVC", "SMOKE", "SCC"]
numeric_cols = ["Age", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
freq_ordinal_cols = ["CAEC", "CALC"]
mtrans_col = "MTRANS"
gender_col = "Gender"
target_col = "NObeyesdad"

yn_map = {"no": 0, "yes": 1}
gender_map = {"male": 0, "female": 1}
freq_map = {"no": 0, "sometimes": 1, "frequently": 2, "always": 3}
mtrans_map = {"walking": 0, "bike": 1, "motorbike": 2, "automobile": 3, "public_transportation": 4}
target_map = {
    "insufficient_weight": 0,
    "normal_weight": 1,
    "overweight_level_i": 2,
    "overweight_level_ii": 3,
    "obesity_type_i": 4,
    "obesity_type_ii": 5,
    "obesity_type_iii": 6,
}

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

df = pd.read_csv(IN_PATH, encoding="utf-8")

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

if target_col in df.columns:
    df[target_col] = df[target_col].astype(str).str.strip().str.lower().map(target_map).astype("Int64")

for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

if "Height" in df.columns:
    df["Height"] = pd.to_numeric(df["Height"], errors="coerce").astype("float64")
if "Weight" in df.columns:
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce").astype("float64")

if "Age" in df.columns:
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce").round().astype("Int64")

anomaly_records = []
for col, (lo, hi) in ranges.items():
    if col in df.columns:
        clip_and_record(df, col, lo, hi, anomaly_records)
for col, (lo, hi) in ordinal_ranges.items():
    if col in df.columns:
        clip_and_record(df, col, lo, hi, anomaly_records)

if "Age" in df.columns:
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce").round().astype("Int64")

for c in ["FCVC", "NCP", "CH2O", "FAF", "TUE", "Height", "Weight"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

for c in binary_cols + freq_ordinal_cols + [gender_col, mtrans_col, target_col]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

columns_to_keep = ["Gender", "FCVC", "SMOKE", "CALC", "NCP", "CH2O", target_col]
df_out = df[columns_to_keep].copy()

for c in ["FCVC", "NCP", "CH2O"]:
    if c in df_out.columns:
        df_out[c] = pd.to_numeric(df_out[c], errors="coerce").astype("float64")
for c in ["Gender", "SMOKE", "CALC"]:
    if c in df_out.columns:
        df_out[c] = pd.to_numeric(df_out[c], errors="coerce").astype("Int64")

float_cols = df_out.select_dtypes(include="float").columns
if len(float_cols) > 0:
    df_out[float_cols] = df_out[float_cols].apply(lambda col: col.map(_fmt_float_str))

df_out.to_csv(OUT_PATH, index=False, na_rep="")

if anomaly_records:
    pd.DataFrame(anomaly_records).to_csv(ANOMALY_REPORT_PATH, index=False)
