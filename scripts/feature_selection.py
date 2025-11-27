import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler

df = pd.read_csv("../data/normalized_minmax_all.csv")

# paliekam tik 4 ir 5 klases
df = df[df['NObeyesdad'].isin([4, 5])]

# pašalinam Weight ir Height iš rinkinio
all_features = [c for c in df.columns if c not in ["NObeyesdad","Weight","Height", "Age", "Gender"]]

X = df[all_features]
y = df["NObeyesdad"]

# 4. Train/test padalinimas (tinkamas etapas požymių atrankai)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Modelis požymių atrankai
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

importances = pd.DataFrame({
    "feature": all_features,
    "importance": dt.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\n=== POŽYMIŲ SVARBOS BE WEIGHT/HEIGHT ===")
print(importances)

selected = ["FCVC", "FAF", "CH2O", "NCP", "TUE", "MTRANS"]

new_data = df[selected + ["NObeyesdad"]]
new_data.to_csv("../data/classification_data.csv", index=False)

print("Sukurtas classification_data.csv")
