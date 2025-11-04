import pandas as pd
import numpy as np

IN_PATH = "./data/data.csv"
OUT_PATH = "./data/clean_data_full.csv"
ANOMALY_REPORT_PATH = "./data/anomalies_report_full.csv"
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

# Visi kategoriški stulpeliai
binary_cols = ["family_history_with_overweight", "FAVC", "SMOKE", "SCC"]
numeric_cols = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
freq_ordinal_cols = ["CAEC", "CALC"]
mtrans_col = "MTRANS"
gender_col = "Gender"
target_col = "NObeyesdad"

# Mapping dictionaries
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

# Diapazony visiems skaitiniams stulpeliams
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

print("Duomenų valymo procesas pradėtas...")
df = pd.read_csv(IN_PATH, encoding="utf-8")
print(f"Užkrauta {len(df)} eilučių, {len(df.columns)} stulpelių")

# Konvertuojame binarinius stulpelius
for c in binary_cols:
    if c in df.columns:
        df[c] = map_values(df[c], yn_map)
        print(f"Konvertuotas binarinis stulpelis: {c}")

# Konvertuojame lytį
if gender_col in df.columns:
    df[gender_col] = map_values(df[gender_col], gender_map)
    print(f"Konvertuotas: {gender_col}")

# Konvertuojame dažnio ordinalinius stulpelius
for c in freq_ordinal_cols:
    if c in df.columns:
        df[c] = map_values(df[c], freq_map)
        print(f"Konvertuotas ordinalinis stulpelis: {c}")

# Konvertuojame transporto būdą
if mtrans_col in df.columns:
    df[mtrans_col] = map_values(df[mtrans_col], mtrans_map)
    print(f"Konvertuotas: {mtrans_col}")

# Konvertuojame target kintamąjį
if target_col in df.columns:
    df[target_col] = df[target_col].astype(str).str.strip().str.lower().map(target_map).astype("Int64")
    print(f"Konvertuotas target stulpelis: {target_col}")

# Konvertuojame skaitinius stulpelius
for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if c == "Age":
            df[c] = df[c].round().astype("Int64")
        else:
            df[c] = df[c].astype("float64")
        print(f"Konvertuotas skaitinis stulpelis: {c}")

# Anomalijų įrašymas ir Winsorizing
anomaly_records = []

# Tvarkome skaitinius stulpelius
for col, (lo, hi) in ranges.items():
    if col in df.columns:
        clip_and_record(df, col, lo, hi, anomaly_records)
        print(f"Patikrinta ir sutvarkyti išskirtys stulpelyje: {col}")

# Tvarkome ordinalinius stulpelius
for col, (lo, hi) in ordinal_ranges.items():
    if col in df.columns:
        clip_and_record(df, col, lo, hi, anomaly_records)
        print(f"Patikrinta ir sutvarkyti išskirtys stulpelyje: {col}")

# Galutinis duomenų tipo nustatymas
if "Age" in df.columns:
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce").round().astype("Int64")

for c in ["Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

for c in binary_cols + freq_ordinal_cols + [gender_col, mtrans_col, target_col]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

# IŠLAIKOME VISUS STULPELIUS (ne tik Chi² atrinktus)
columns_to_keep = []
for col in df.columns:
    columns_to_keep.append(col)

df_out = df[columns_to_keep].copy()

# Formatuojame float stulpelius
float_cols = df_out.select_dtypes(include="float").columns
if len(float_cols) > 0:
    df_out[float_cols] = df_out[float_cols].apply(lambda col: col.map(_fmt_float_str))

print(f"\nIšlaikyta {len(columns_to_keep)} stulpelių:")
print(f"Stulpeliai: {list(columns_to_keep)}")

# Išsaugome duomenis
df_out.to_csv(OUT_PATH, index=False, na_rep="")
print(f"Išvalyti duomenys išsaugoti: {OUT_PATH}")

# Išsaugome anomalijų ataskaitą
if anomaly_records:
    pd.DataFrame(anomaly_records).to_csv(ANOMALY_REPORT_PATH, index=False)
    print(f"Rasta ir sutvarkyti {len(anomaly_records)} anomalijų")
    print(f"Anomalijų ataskaita išsaugota: {ANOMALY_REPORT_PATH}")
else:
    print("Anomalijų nerasta")

print(f"\nGalutinė duomenų statistika:")
print(f"Eilučių skaičius: {len(df_out)}")
print(f"Stulpelių skaičius: {len(df_out.columns)}")
print(f"Trūkstamų reikšmių: {df_out.isnull().sum().sum()}")

print("\nStulpelių tipai:")
for col in df_out.columns:
    print(f"  {col}: {df_out[col].dtype}")

print("\nDuomenų valymo procesas baigtas!")