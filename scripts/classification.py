import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
import os
import warnings
warnings.filterwarnings('ignore')

print("=== KLASIFIKACIJOS MODELIŲ ANALIZĖ ===")

# 1. Duomenų užkrovimas
try:
    df = pd.read_csv('./data/normalized_minmax_all.csv')
    print(f"Duomenys užkrauti sėkmingai: {df.shape}")
    print(f"Stulpeliai: {list(df.columns)}")
except FileNotFoundError:
    print("Klaida: Nepavyko rasti failo './data/normalized_minmax_all.csv'")
    exit()

# FIKSUOTAS TARGET STULPELIO PAVADINIMAS
X = df.drop('NObeyesdad', axis=1)  # ← FIKSUOTAS: 'NObeyesdad' vietoj 'target'
y = df['NObeyesdad']

print(f"Požymiai (X): {X.shape}")
print(f"Tikslai (y): {y.shape}")
print(f"Klasės: {sorted(y.unique())}")
print(f"Klasių pasiskirstymas:")
print(y.value_counts().sort_index())

# 2. Skaidymas (pridėtas stratify multiclass duomenims)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTreniravimo rinkinys: {X_train.shape}")
print(f"Testavimo rinkinys: {X_test.shape}")

# 3. Skalavimas (svarbu K-NN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Modelių apibrėžimas
models = {
    "K-NN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# 5. Ciklas per modelius rezultatams gauti
results = {}
n_classes = len(np.unique(y))
print(f"\nRasta {n_classes} klasių: {sorted(np.unique(y))}")

for name, model in models.items():
    print(f"\n=== {name.upper()} MODELIO TRENIRAVIMAS ===")
    
    # Treniruojame modelį
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # Tikslumas
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # MULTICLASS ROC KREIVĖS (One-vs-Rest)
    try:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)
            
            # Skaičiuojame AUC kiekvienai klasei (One-vs-Rest)
            fpr = {}
            tpr = {}
            roc_auc = {}
            
            for i, class_label in enumerate(sorted(np.unique(y))):
                # Binariška klasifikacija: klasė i prieš likusias
                y_binary = (y_test == class_label).astype(int)
                
                if len(np.unique(y_binary)) > 1 and i < y_prob.shape[1]:
                    fpr[class_label], tpr[class_label], _ = roc_curve(y_binary, y_prob[:, i])
                    roc_auc[class_label] = auc(fpr[class_label], tpr[class_label])
            
            # Vidutinis AUC (macro-average)
            avg_auc = np.mean(list(roc_auc.values())) if roc_auc else 0.0
        else:
            fpr, tpr, roc_auc, avg_auc = {}, {}, {}, 0.0
    except Exception as e:
        print(f"ROC skaičiavimo klaida: {e}")
        fpr, tpr, roc_auc, avg_auc = {}, {}, {}, 0.0
    
    results[name] = {
        "Accuracy": acc,
        "CM": cm,
        "FPR": fpr,
        "TPR": tpr,
        "ROC_AUC": roc_auc,
        "Avg_AUC": avg_auc
    }
    
    print(f"Tikslumas: {acc:.4f}")
    print(f"Vidutinis AUC: {avg_auc:.4f}")
    print("\nKlasifikacijos ataskaita:")
    print(classification_report(y_test, y_pred, zero_division=0))

# 6. Vizualizacijos kūrimas
os.makedirs('outputs/classification', exist_ok=True)

# 6.1 Palyginimo suvestinė
print(f"\n{'='*50}")
print(f"{'MODELIŲ PALYGINIMO SUVESTINĖ':^50}")
print(f"{'='*50}")

comparison_data = []
for name, res in results.items():
    comparison_data.append({
        'Modelis': name,
        'Tikslumas': f"{res['Accuracy']:.4f}",
        'Vid. AUC': f"{res['Avg_AUC']:.4f}"
    })

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# 6.2 Vizualizacijos
fig = plt.figure(figsize=(16, 12))

# Subplot 1: Tikslumų palyginimas
plt.subplot(2, 3, 1)
names = list(results.keys())
accuracies = [results[name]['Accuracy'] for name in names]
colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']

bars = plt.bar(names, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
plt.title('Modelių tikslumas', fontweight='bold', fontsize=14)
plt.ylabel('Tikslumas')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.ylim(0, 1)

# Pridedame reikšmes ant stulpelių
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

# Subplot 2: AUC palyginimas
plt.subplot(2, 3, 2)
aucs = [results[name]['Avg_AUC'] for name in names]

bars = plt.bar(names, aucs, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
plt.title('Vidutinis AUC (Macro)', fontweight='bold', fontsize=14)
plt.ylabel('AUC')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.ylim(0, 1)

for bar, auc_val in zip(bars, aucs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{auc_val:.3f}', ha='center', va='bottom', fontweight='bold')

# Subplot 3: Cross-Validation rezultatai
plt.subplot(2, 3, 3)
cv_results = []
cv_names = []

print(f"\n{'='*40}")
print(f"{'CROSS-VALIDATION ANALIZĖ':^40}")
print(f"{'='*40}")

for name, model in models.items():
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kfold, scoring='accuracy')
    cv_results.append(cv_scores)
    cv_names.append(name)
    
    print(f"{name}:")
    print(f"  Vidutinis CV: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
    print(f"  Min-Max: {cv_scores.min():.4f} - {cv_scores.max():.4f}")

box_plot = plt.boxplot(cv_results, labels=cv_names, patch_artist=True)

# Spalviname dėžes
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)

plt.title('Cross-Validation (5-fold)', fontweight='bold', fontsize=14)
plt.ylabel('Tikslumas')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# Subplot 4-6: Confusion Matrices (3 geriausių)
sorted_results = sorted(results.items(), key=lambda x: x[1]['Accuracy'], reverse=True)

for i, (name, result) in enumerate(sorted_results[:3]):
    plt.subplot(2, 3, 4 + i)
    cm = result['CM']
    
    # Sukuriame heatmap
    im = plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f'{name}\n(Acc: {result["Accuracy"]:.3f})', fontweight='bold')
    
    # Pridedame tekstą kiekvienai ląstelei
    thresh = cm.max() / 2.
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            plt.text(col, row, f'{cm[row, col]}',
                    ha="center", va="center", fontweight='bold',
                    color="white" if cm[row, col] > thresh else "black")
    
    plt.xlabel('Prognozuota klasė')
    plt.ylabel('Tikra klasė')
    
    # Ašių žymės
    classes = sorted(np.unique(y))
    plt.xticks(range(len(classes)), classes)
    plt.yticks(range(len(classes)), classes)

plt.tight_layout()
plt.savefig('outputs/classification/classification_comprehensive_analysis.png', 
            dpi=300, bbox_inches='tight')
plt.show()

# 6.3 ROC kreivės (jei yra duomenų)
if any(results[name]['ROC_AUC'] for name in names):
    plt.figure(figsize=(15, 10))
    
    # Sukuriame subplot kiekvienai klasei
    n_classes = len(np.unique(y))
    cols = min(3, n_classes)
    rows = (n_classes + cols - 1) // cols
    
    for class_idx, class_label in enumerate(sorted(np.unique(y))):
        plt.subplot(rows, cols, class_idx + 1)
        
        for name in names:
            if class_label in results[name]['ROC_AUC']:
                fpr = results[name]['FPR'][class_label]
                tpr = results[name]['TPR'][class_label]
                auc_val = results[name]['ROC_AUC'][class_label]
                plt.plot(fpr, tpr, label=f'{name} (AUC={auc_val:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Kreivės - Klasė {class_label}', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/classification/roc_curves_by_class.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

print(f"\n{'='*50}")
print(f"{'ANALIZĖS SUVESTINĖ':^50}")
print(f"{'='*50}")

print("Vizualizacijos išsaugotos:")
print(" outputs/classification/classification_comprehensive_analysis.png")
if any(results[name]['ROC_AUC'] for name in names):
    print("  outputs/classification/roc_curves_by_class.png")

best_model = max(results.keys(), key=lambda x: results[x]['Accuracy'])
best_accuracy = results[best_model]['Accuracy']
print(f"\nGERIAUSIAS MODELIS: {best_model}")
print(f"   Tikslumas: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")

print(f"\nVISŲ MODELIŲ REITINGAS:")
for i, (name, result) in enumerate(sorted_results):
    print(f"   {i+1}. {name}: {result['Accuracy']:.4f} ({result['Accuracy']*100:.2f}%)")

print("\n Analizė baigta sėkmingai!")