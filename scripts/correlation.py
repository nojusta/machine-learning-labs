import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

FILE_PATH = "../data/clean_data.csv"  # or the non-normalized one if you want

df = pd.read_csv(FILE_PATH)

# Make sure we're only using classes 4–6 (obesity types I–III)
df = df[df["NObeyesdad"].isin([4, 5, 6])]

# Select features (everything except the label)
feature_cols = [c for c in df.columns if c != "NObeyesdad"]

# Convert to numeric just to be safe
for c in feature_cols + ["NObeyesdad"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# --- Pearson correlation matrix ---
corr_matrix = df[feature_cols + ["NObeyesdad"]].corr(method="pearson")
print("=== Correlation matrix ===")
print(corr_matrix)

# --- Feature-class correlations ---
print("\n=== Correlation with class (NObeyesdad) ===")
print(corr_matrix["NObeyesdad"].sort_values(ascending=False))

# --- Heatmap ---
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap (Pearson)")
plt.tight_layout()
plt.show()

grouped_var = df.groupby("NObeyesdad").var()
print(grouped_var)

print(df[df["NObeyesdad"] == 6].nunique())
