import pandas as pd

df = pd.read_csv('../data/clean_data.csv')

selected_columns = ['Units_Sold', 'Sales', 'Discounts', 'Sale_Price', 'Gross_Sales']
df_selected = df[selected_columns].copy()

numeric_cols = ['Units_Sold', 'Sales', 'Discounts', 'Sale_Price', 'Gross_Sales']
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

print("\nAprašomosios statistikos lentelė prieš atsiskyrėlius:")
print(stats)

def find_outliers(df, columns):
    outliers = {}
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
    return outliers

outliers = find_outliers(df_selected, numeric_cols)

print("\nTaškai atsiskyrėliai:")
for col, vals in outliers.items():
    print(f"{col}: {vals.tolist()}")


df_no_outliers = df_selected.copy()
for col, vals in outliers.items():
    df_no_outliers = df_no_outliers[~df_no_outliers[col].isin(vals)]

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

print("\nAprašomosios statistikos lentelė po atsiskyrėlių pašalinimo:")
print(stats_no_outliers)
