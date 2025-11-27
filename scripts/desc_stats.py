import pandas as pd

# Kad nieko netrumpintų vaizde
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# 1. Nuskaitom duomenis
df = pd.read_csv("../data/normalized_minmax_all.csv")

# 2. Klasės stulpelio pavadinimas
class_col = "NObeyesdad"  # pakeisk, jei vadinasi kitaip

# 3. Filterinam tik 4 ir 5 klases
df_bin = df[df[class_col].isin([4, 5])].copy()

# 4. Randam visus skaitinius požymius, išmetam klasę
numeric_cols = df_bin.select_dtypes(include=["int64", "float64"]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != class_col]

print("Naudojami skaitiniai požymiai:")
print(numeric_cols)
print()

# 5. Funkcija, kuri vienai klasei padaro tvarkingą lentelę:
def get_stats_for_class(data, cls_value):
    sub = data[data[class_col] == cls_value][numeric_cols]
    stats = sub.agg(["min", "max", "mean", "median", "std"]).T
    stats = stats.round(4)
    # Pervadinam stulpelius į LT
    stats.columns = ["Min", "Max", "Vidurkis", "Mediana", "Std"]
    stats.index.name = "Požymis"
    return stats

stats_4 = get_stats_for_class(df_bin, 4)
stats_5 = get_stats_for_class(df_bin, 5)

print("4 klasės (Obesity Type I) aprašomoji statistika:\n")
print(stats_4.to_string())
print("\n" + "="*80 + "\n")
print("5 klasės (Obesity Type II) aprašomoji statistika:\n")
print(stats_5.to_string())
