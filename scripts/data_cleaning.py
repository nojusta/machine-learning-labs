import pandas as pd
import numpy as np
import re

IN_PATH = "../data/data.csv"
OUT_PATH = "../data/clean_data.csv"

def clean_colnames(cols):
    return [re.sub(r"\s+", "_", c.strip()) for c in cols]

def parse_currency(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s == "$-":
        return 0.0
    s = s.replace("$", "").replace(",", "").strip()
    if s in ("", "-"):
        return np.nan
    try:
        return float(s)
    except:
        return np.nan

df = pd.read_csv(IN_PATH, encoding="utf-8")
df.columns = clean_colnames(df.columns)

money_cols = [
    "Manufacturing_Price", "Sale_Price", "Gross_Sales",
    "Discounts", "Sales", "COGS", "Profit"
]
for c in money_cols:
    df[c] = df[c].apply(parse_currency)

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().mean() > 0.9:  
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    parsed = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"] = parsed.dt.year.where(~parsed.isna(), df.get("Year"))
    df["Month_Number"] = parsed.dt.month.where(~parsed.isna(), df.get("Month_Number"))

if {"Gross_Sales", "Sale_Price"}.issubset(df.columns):
    denom = df["Sale_Price"].replace(0, np.nan)
    units_float = df["Gross_Sales"] / denom
    def round_half_up(x):
        if pd.isna(x):
            return np.nan
        return int(np.floor(x + 0.5))
    df["Units_Sold"] = units_float.apply(round_half_up).astype("Int64")

df.to_csv(OUT_PATH, index=False)
print(f"Cleaned file saved to {OUT_PATH}")
