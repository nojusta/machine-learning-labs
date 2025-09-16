import pandas as pd

df = pd.read_csv('../data/data.csv')
df.columns = df.columns.str.strip()

selected_columns = ['Units Sold', 'Sales', 'Discounts']
df_selected = df[selected_columns].copy()

for col in ['Units Sold', 'Sales', 'Discounts']:
    df_selected[col] = df_selected[col].replace(r'^\s*\$?-+\s*$', '0', regex=True)
    df_selected[col] = df_selected[col].replace(r'^\s*$', '0', regex=True)
    df_selected[col] = df_selected[col].replace(r'[\$,]', '', regex=True)
    df_selected[col] = df_selected[col].astype(float)

df_selected['Units Sold'] = df_selected['Units Sold'].round().astype(int)

numeric_cols = ['Units Sold', 'Sales', 'Discounts']
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
    if x == int(x):
        return f"{int(x):,}"
    else:
        return f"{x:,.2f}"

for col in numeric_cols:
    stats[col] = stats[col].apply(format_number)

print("\nAprašomosios statistikos lentelė:")
print(stats)