import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Įkeliame sunormuotus duomenis
df_standardized = pd.read_csv("./data/normalized_minmax.csv")   #<-- Pakeitus failus tarp minmax ir mean_var
                                                                #gaunam lenteles po normalizavicjos ir outlier panaikinimo
numeric_cols = df_standardized.columns

# Histogramos kiekvienam požymiui
plt.figure(figsize=(12, 8))
for i, col in enumerate(numeric_cols):
    plt.subplot(2, 3, i+1)
    plt.hist(df_standardized[col], bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Stačiakampės diagramos (Boxplot) kiekvienam požymiui
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_standardized)
plt.title('Boxplot of standardized features')
plt.xticks(rotation=45)
plt.show()

# Taškiniai grafikai (scatter plot matrix)
sns.pairplot(df_standardized)
plt.suptitle("Scatter plot matrix of standardized features", y=1.02)
plt.show()

# Šilumos žemėlapis koreliacijai (reikalinga tolimesnei analizei)
plt.figure(figsize=(8,6))
sns.heatmap(df_standardized.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation heatmap of standardized features")
plt.show()
