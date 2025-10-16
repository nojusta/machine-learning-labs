import pandas as pd
import numpy as np

# Nuskaitymas
df = pd.read_csv('../data/normalized_minmax.csv')

# Stulpeliai, pagal kuriuos tikrinamos išskirtys
features = ["Gender", "FCVC", "SMOKE", "CALC", "NCP", "CH2O"]

# Inicializuojam žymes
inner_mask = np.zeros(len(df), dtype=bool)
outer_mask = np.zeros(len(df), dtype=bool)

for col in features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    # Ribos
    inner_lower = Q1 - 1.5 * IQR
    inner_upper = Q3 + 1.5 * IQR
    outer_lower = Q1 - 3 * IQR
    outer_upper = Q3 + 3 * IQR

    # Žymim inner ir outer
    inner_mask |= ((df[col] < inner_lower) | (df[col] > inner_upper))
    outer_mask |= ((df[col] < outer_lower) | (df[col] > outer_upper))

# Sukuriam stulpelį su reikšmėmis:
# 0 - normalus
# 1 - vidinė (inner)
# 2 - išorinė (outer)
df["outlier_type"] = 0
df.loc[inner_mask, "outlier_type"] = 1
df.loc[outer_mask, "outlier_type"] = 2

# Skaičiuojam rezultatus
inner_count = sum(df["outlier_type"] == 1)
outer_count = sum(df["outlier_type"] == 2)
total_count = sum(df["outlier_type"] != 0)

# Spausdinam statistiką
print("----- Išskirčių statistika -----")
print(f"Vidinių (inner) išskirčių: {inner_count}")
print(f"Išorinių (outer) išskirčių: {outer_count}")
print(f"Iš viso išskirčių: {total_count}")

# Išsaugom į naują CSV
df.to_csv("../data/outliers.csv", index=False)
print("\nRezultatai išsaugoti faile: normalized_selected_features_with_outlier_types.csv")
