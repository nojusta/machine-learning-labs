import pandas as pd

IN_PATH = "./data/data.csv"
OUT_PATH = "./data/clean_data.csv"

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
numeric_cols = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
freq_ordinal_cols = ["CAEC", "CALC"]  # no < Sometimes < Frequently < Always
# Transport is nominal; we assign integer labels (not ordinal)
mtrans_col = "MTRANS"
gender_col = "Gender"
target_col = "NObeyesdad"

# Mappings
yn_map = {"no": 0, "yes": 1}
gender_map = {"male": 0, "female": 1}
freq_map = {"no": 0, "sometimes": 1, "frequently": 2, "always": 3}
mtrans_map = {
    "public_transportation": 0,
    "automobile": 1,
    "walking": 2,
    "bike": 3,
    "motorbike": 4,
}
# Target label order (light -> heavy)
target_map = {
    "insufficient_weight": 0,
    "normal_weight": 1,
    "overweight_level_i": 2,
    "overweight_level_ii": 3,
    "obesity_type_i": 4,
    "obesity_type_ii": 5,
    "obesity_type_iii": 6,
}

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

if target_col in df.columns:
    df[target_col] = (
        df[target_col].astype(str).str.strip().str.lower().map(target_map).astype("Int64")
    )

df.to_csv(OUT_PATH, index=False)
print(f"Clean data saved to {OUT_PATH}")
print(df.head())
print(df.dtypes)