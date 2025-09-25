import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt


df = pd.read_csv('../data/clean_data.csv')

selected_columns = ['Units_Sold', 'Sales', 'Discounts', 'Sale_Price', 'Gross_Sales']
df_selected = df[selected_columns].copy()
numeric_cols = selected_columns

stats = df_selected[numeric_cols].agg(['min', 'max', 'mean', 'median', 'var'])
stats.loc['1 kvartilė'] = df_selected[numeric_cols].quantile(0.25)
stats.loc['3 kvartilė'] = df_selected[numeric_cols].quantile(0.75)
stats = stats.rename(index={
    'min': 'Min',
    'max': 'Max',
    'mean': 'Vidurkis',
    'median': 'Mediana',
    'var': 'Dispersija'
})

def format_number(x):
    if pd.isna(x):
        return "NaN"
    elif x == int(x):
        return f"{int(x):,}"
    else:
        return f"{x:,.2f}"

for col in numeric_cols:
    stats[col] = stats[col].apply(format_number)

print("\nAprašomosios statistikos lentelė prieš atmetant išskirtis:")
print(stats)

def find_outliers(df, columns):
    outliers = {}
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_outer = Q1 - 3 * IQR
        upper_outer = Q3 + 3 * IQR

        extreme = df[(df[col] < lower_outer) | (df[col] > upper_outer)][col]

        outliers[col] = extreme.tolist()
    return outliers

outliers = find_outliers(df_selected, numeric_cols)

print("\nIšimtys:")
for col, vals in outliers.items():
    print(f"{col}: {vals}")

print("\nIšimtys:")
for col, vals in outliers.items():
    print(f"{col}: {len(vals)} reikšmės")


df_no_outliers = df_selected.copy()
for col, vals in outliers.items():
    df_no_outliers = df_no_outliers[~df_no_outliers[col].isin(vals)]

df_no_outliers.to_csv('../data/clean_data_without_outliers.csv', index=False)

stats_no_outliers = df_no_outliers[numeric_cols].agg(['min', 'max', 'mean', 'median', 'var'])
stats_no_outliers.loc['1 kvartilė'] = df_no_outliers[numeric_cols].quantile(0.25)
stats_no_outliers.loc['3 kvartilė'] = df_no_outliers[numeric_cols].quantile(0.75)
stats_no_outliers = stats_no_outliers.rename(index={
    'min': 'Min',
    'max': 'Max',
    'mean': 'Vidurkis',
    'median': 'Mediana',
    'var': 'Dispersija'
})

for col in numeric_cols:
    stats_no_outliers[col] = stats_no_outliers[col].apply(format_number)

print("\nAprašomosios statistikos lentelė po išimčių pašalinimo:")
print(stats_no_outliers)


means = df_no_outliers.mean()
stds = df_no_outliers.std(ddof=1)
df_standardized = (df_no_outliers - means) / stds

print("\nSunormuoti duomenys (pagal vidurkį ir dispersiją):")
print(df_standardized.head())


mins = df_no_outliers.min()
maxs = df_no_outliers.max()
df_minmax = (df_no_outliers - mins) / (maxs - mins)

print("\nSunormuoti duomenys (Min-Max [0,1]):")
print(df_minmax.head())


df_standardized.to_csv("../data/normalized_mean_var.csv", index=False)
df_minmax.to_csv("../data/normalized_minmax.csv", index=False)

print("\nSunormuota visa duomenų aibė. Rezultatai įrašyti į failus:")
print(" - normalized_mean_var.csv (Vidurkis ir dispersija)")
print(" - normalized_minmax.csv (Min-Max)")
print(" - clean_data_without_outliers.csv (be ekstremalių atsiskyrėlių)")


def millions(x, pos):
    return f'{x/1e6:.1f}M'

formatter = FuncFormatter(millions)

product_summary = df.groupby('Product').agg({
    'Units_Sold': 'sum',
    'Sales': 'sum',
    'Discounts': 'sum'
})
product_summary['Average_Price'] = product_summary['Sales'] / product_summary['Units_Sold']
product_summary = product_summary.sort_values(by='Units_Sold', ascending=False)

print("\nBendri pardavimai pagal produktus:\n")
print(product_summary)

totals = product_summary.sum(numeric_only=True)
print("\nBendros sumos visiems produktams:")
print(totals)

plt.figure(figsize=(16, 12))
sns.set(style="whitegrid")

plt.subplot(2, 2, 1)
sns.barplot(x=product_summary.index, y=product_summary['Sales'], palette="magma")
plt.title("Pajamos pagal produktus", fontsize=14)
plt.ylabel("Sales (M EUR)")
plt.gca().yaxis.set_major_formatter(formatter)
plt.xticks(rotation=45)

plt.subplot(2, 2, 2)
sns.barplot(x=product_summary.index, y=product_summary['Units_Sold'], palette="viridis")
plt.title("Parduotų vienetų kiekis", fontsize=14)
plt.ylabel("Units Sold")
plt.xticks(rotation=45)

plt.subplot(2, 2, 3)
sns.barplot(x=product_summary.index, y=product_summary['Discounts'], palette="cubehelix")
plt.title("Nuolaidos pagal produktus", fontsize=14)
plt.ylabel("Discounts (M EUR)")
plt.gca().yaxis.set_major_formatter(formatter)
plt.xticks(rotation=45)

plt.subplot(2, 2, 4)
sns.barplot(x=product_summary.index, y=product_summary['Average_Price'], palette="coolwarm")
plt.title("Vidutinė pardavimo kaina pagal produktus", fontsize=14)
plt.ylabel("Average Price (EUR)")
plt.xticks(rotation=45)


plt.tight_layout()
plt.show()