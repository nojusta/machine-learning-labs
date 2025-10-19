import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

FILE_PATH = "../data/clean_data.csv"

df = pd.read_csv(FILE_PATH)

df = df[df["NObeyesdad"].isin([4, 5, 6])]

feature_cols = [c for c in df.columns if c != "NObeyesdad"]

for c in feature_cols + ["NObeyesdad"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

corr_matrix = df[feature_cols + ["NObeyesdad"]].corr(method="pearson")
print("=== Correlation matrix ===")
print(corr_matrix)

print("\n=== Correlation with class (NObeyesdad) ===")
print(corr_matrix["NObeyesdad"].sort_values(ascending=False))

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap (Pearson)")
plt.tight_layout()
plt.show()

grouped_var = df.groupby("NObeyesdad").var()
print(grouped_var)

print(df[df["NObeyesdad"] == 6].nunique())
