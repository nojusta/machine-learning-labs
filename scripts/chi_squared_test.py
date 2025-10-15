import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif

df = pd.read_csv("../data/normalized_minmax.csv")

df = df[df['NObeyesdad'].isin([4,5,6])]

X = df.drop(columns=['NObeyesdad'])
y = df['NObeyesdad']

chi_selector = SelectKBest(score_func=chi2, k=6)
chi_selector.fit(X, y)
chi_scores = pd.Series(chi_selector.scores_, index=X.columns).sort_values(ascending=False)
print("Chi² scores:")
print(chi_scores)
