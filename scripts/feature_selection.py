import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

# 1. Nuskaitymas
df = pd.read_csv("../data/normalized_minmax_all.csv")

# 2. Imame tik 4 ir 5 klases
df = df[df['NObeyesdad'].isin([4, 5])]

# 3. Pasirenkame jūsų atrinktus požymius
selected_features = ["Weight", "Height", "FCVC", "Age", "CH2O", "CAEC"]

X = df[selected_features]
y = df["NObeyesdad"]

# 4. Train/test padalinimas
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Skalavimas tik train (teisingai)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Decision Tree požymių svarba
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_scaled, y_train)

importances = dt.feature_importances_
feature_importance = pd.DataFrame({
    "feature": selected_features,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print(feature_importance)

output_df = df[selected_features + ["NObeyesdad"]]
output_df.to_csv("../data/selected_features.csv", index=False)

print("\nFailas 'selected_features.csv' sėkmingai sukurtas!")