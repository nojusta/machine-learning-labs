import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler


# 1) Load + keep only class 4 & 5
df = pd.read_csv("../data/normalized_minmax_all.csv")
df = df[df['NObeyesdad'].isin([4, 5])]

print("\n=== Pradinis klasių kiekis (prieš balansavimą) ===")
print(df["NObeyesdad"].value_counts())


# 2) Feature selection (same as before)
all_features = [c for c in df.columns if c not in ["NObeyesdad","Weight","Height", "Age", "Gender"]]
X = df[all_features]
y = df["NObeyesdad"]

# Train/test split tikai atributų atrankai, ne balansui
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Feature importance from Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

importances = pd.DataFrame({
    "feature": all_features,
    "importance": dt.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\n=== POŽYMIŲ SVARBOS ")
print(importances)

# Naudojami atrinkti požymiai
selected = ["FCVC", "FAF", "CH2O", "NCP", "TUE", "MTRANS"]
df_selected = df[selected + ["NObeyesdad"]]


#UNDERSAMPLING — BALANSAVIMAS
rus = RandomUnderSampler(random_state=42)
X_bal, y_bal = rus.fit_resample(df_selected[selected], df_selected["NObeyesdad"])

df_balanced = pd.DataFrame(X_bal, columns=selected)
df_balanced["NObeyesdad"] = y_bal

print("\n=== Po undersampling balanso (bus 1:1) ===")
print(df_balanced["NObeyesdad"].value_counts())

df_balanced.to_csv("../data/classification_data.csv", index=False)

print("\nSukurtas SUBALANSUOTAS classification_data.csv")
print(f"Iš viso įrašų → {len(df_balanced)}")

# 80/20 padalijimas mokymui+validacijai ir testavimui
train_val_df, test_df = train_test_split(
    df_balanced,
    test_size=0.2,
    stratify=df_balanced["NObeyesdad"],
    random_state=42,
)

train_val_df.to_csv("../data/classification_train_val.csv", index=False)
test_df.to_csv("../data/classification_test.csv", index=False)

print("\nSukurtas SUBALANSUOTAS classification_data.csv")
print(f"Iš viso įrašų: {len(df_balanced)}")
print(f"Train+Val: {len(train_val_df)}, Test: {len(test_df)}")