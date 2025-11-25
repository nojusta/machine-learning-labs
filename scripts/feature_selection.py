import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler

# 1. Nuskaitymas
df = pd.read_csv("../data/normalized_minmax_all.csv")

# 2. Imame tik 4 ir 5 klases
df = df[df['NObeyesdad'].isin([4, 5])]

# 3. Pasirenkame jūsų atrinktus požymius
selected_features = ["Weight", "Height", "FCVC", "Age", "CH2O", "CAEC"]

X = df[selected_features]
y = df["NObeyesdad"]

# 4. Train/test padalinimas (tinkamas etapas požymių atrankai)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Skalavimas (nebūtinas DT, bet čia paliktas jei vėliau naudosi kitus modelius)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Decision Tree požymių svarba (tik požymių atrankai)
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_scaled, y_train)

importances = dt.feature_importances_
feature_importance = pd.DataFrame({
    "feature": selected_features,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print(feature_importance)

# 7. Sukuriame failą su atrinktais požymiais (darbo 4.1 daliai)
output_df = df[selected_features + ["NObeyesdad"]]
output_df.to_csv("../data/selected_features.csv", index=False)

print("\nFailas 'selected_features.csv' sėkmingai sukurtas!")

# 8. DUOMENŲ SUBALANSAVIMAS (Random Undersampling)
rus = RandomUnderSampler(random_state=42)
X_bal, y_bal = rus.fit_resample(X, y)

balanced_df = pd.concat([
    pd.DataFrame(X_bal, columns=selected_features),
    pd.Series(y_bal, name="NObeyesdad")
], axis=1)

# 9. Sukuriame classification_data.csv klasifikavimo tyrimui
balanced_df.to_csv("../data/balanced_data.csv", index=False)

print("Failas 'classification_data.csv' sėkmingai sukurtas (su undersampling)!")
