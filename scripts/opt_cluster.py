import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(use_all_features=False):
    """
    Duomenų nuskaitymas ir paruošimas
    """
    print("=== DUOMENŲ PARUOŠIMAS ===")
    
    if use_all_features:
        # Naudojame visus požymius
        filename = "normalized_minmax_all.csv"
        print("Naudojami visi požymiai")
    else:
        # Naudojame Chi^2 atrinktus požymius
        filename = "normalized_minmax.csv"
        print("Naudojami Chi^2 atrinkti požymiai")
    
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "data", filename),
        os.path.join(".", "data", filename),
        f"../data/{filename}"
    ]
    
    df = None
    for path in candidates:
        if os.path.isfile(path):
            df = pd.read_csv(path)
            print(f"Užkrautas failas: {path}")
            break
    
    if df is None:
        raise FileNotFoundError(f"Nerastas {filename} failas")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Pašaliname target kintamąjį jei egzistuoja
    if 'NObeyesdad' in numeric_cols:
        numeric_cols = numeric_cols.drop('NObeyesdad')
    
    X = df[numeric_cols].values
    
    print(f"Duomenų forma: {X.shape}")
    print(f"Naudojami požymiai: {list(numeric_cols)}")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, df, numeric_cols

def empirical_method(X):
    """
    Empirinis metodas: k = sqrt(n/2)
    """
    print("\n=== EMPIRINIS METODAS ===")
    
    n = X.shape[0]
    k_empirical = int(np.sqrt(n / 2))
    
    print(f"Taškų skaičius (n): {n}")
    print(f"Empirinis k = sqrt(n/2) = sqrt({n}/2) = {k_empirical}")
    
    return k_empirical

def elbow_method(X, max_k=10):
    """
    Elbow metodas - SSE skaičiavimas
    """
    print("\n=== ELBOW METODAS ===")
    
    K = range(2, max_k + 1)
    sse = []
    
    print("Skaičiuojami SSE rezultatai:")
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        kmeans.fit(X)
        sse_value = kmeans.inertia_
        sse.append(sse_value)
        print(f"  k={k}: SSE = {sse_value:.2f}")
    
    # Automatinis alkūnės nustatymas
    differences = np.diff(sse)
    second_differences = np.diff(differences)
    elbow_point = np.argmax(second_differences) + 2
    
    if elbow_point >= max_k:
        elbow_point = 3
    
    print(f"Nustatyta alkūnė: k = {elbow_point}")
    
    # Vizualizacija
    plt.figure(figsize=(10, 6))
    plt.plot(K, sse, marker='o', linewidth=2, markersize=8)
    plt.axvline(x=elbow_point, color='red', linestyle='--', alpha=0.7, 
                label=f'Alkūnė: k={elbow_point}')
    plt.xlabel('Klasterių skaičius (k)')
    plt.ylabel('SSE (Within-cluster sum of squares)')
    plt.title('Elbow metodas')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(K)
    
    os.makedirs('outputs', exist_ok=True)
    plt.tight_layout()
    plt.savefig('outputs/elbow_method.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return elbow_point, sse, K

def silhouette_method(X, max_k=10):
    """
    Vidutinio silueto metodas
    """
    print("\n=== VIDUTINIO SILUETO METODAS ===")
    
    K = range(2, max_k + 1)
    sil_scores = []
    
    print("Skaičiuojami silueto koeficientai:")
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        sil_scores.append(score)
        print(f"  k={k}: Silueto koeficientas = {score:.4f}")
    
    max_score_idx = np.argmax(sil_scores)
    optimal_k_silhouette = K[max_score_idx]
    max_score = sil_scores[max_score_idx]
    
    print(f"Maksimalus silueto koeficientas: {max_score:.4f} (k={optimal_k_silhouette})")
    
    # Vizualizacija
    plt.figure(figsize=(10, 6))
    plt.plot(K, sil_scores, marker='o', linewidth=2, markersize=8, color='green')
    plt.axvline(x=optimal_k_silhouette, color='red', linestyle='--', alpha=0.7, 
                label=f'Maksimumas: k={optimal_k_silhouette}')
    plt.xlabel('Klasterių skaičius (k)')
    plt.ylabel('Vidutinis silueto koeficientas')
    plt.title('Silueto metodas')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(K)
    
    plt.tight_layout()
    plt.savefig('outputs/silhouette_method.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return optimal_k_silhouette, sil_scores, K

def combined_analysis_plot(K_range, sse_values, sil_scores, k_emp, k_elbow, k_sil):
    """
    Kombinuotas grafikas
    """
    print("\n=== KOMBINUOTA ANALIZĖ ===")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Elbow grafikas
    ax1.plot(K_range, sse_values, marker='o', linewidth=2, markersize=8, color='blue')
    ax1.axvline(x=k_elbow, color='red', linestyle='--', alpha=0.7, label=f'Elbow: k={k_elbow}')
    ax1.axvline(x=k_emp, color='orange', linestyle=':', alpha=0.7, label=f'Empirinis: k={k_emp}')
    ax1.set_xlabel('Klasterių skaičius (k)')
    ax1.set_ylabel('SSE')
    ax1.set_title('Elbow metodas')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xticks(K_range)
    
    # Silueto grafikas
    ax2.plot(K_range, sil_scores, marker='o', linewidth=2, markersize=8, color='green')
    ax2.axvline(x=k_sil, color='red', linestyle='--', alpha=0.7, label=f'Siluetas: k={k_sil}')
    ax2.axvline(x=k_emp, color='orange', linestyle=':', alpha=0.7, label=f'Empirinis: k={k_emp}')
    ax2.set_xlabel('Klasterių skaičius (k)')
    ax2.set_ylabel('Silueto koeficientas')
    ax2.set_title('Silueto metodas')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xticks(K_range)
    
    plt.tight_layout()
    plt.savefig('outputs/combined_cluster_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_report(k_emp, k_elbow, k_sil, sse_values, sil_scores, X_shape, K_range):
    """
    Ataskaitos generavimas
    """
    print("\n" + "="*60)
    print("           KLASTERIŲ SKAIČIAUS NUSTATYMO ATASKAITA")
    print("="*60)
    
    print(f"\nDUOMENYS:")
    print(f"  Taškų skaičius: {X_shape[0]}")
    print(f"  Požymių skaičius: {X_shape[1]}")
    print(f"  Duomenys normalizuoti: Taip")
    
    print(f"\nMETODŲ REZULTATAI:")
    print(f"  1. Empirinis metodas:     k = {k_emp}")
    print(f"  2. Elbow metodas:         k = {k_elbow}")
    print(f"  3. Silueto metodas:       k = {k_sil}")
    
    min_sse_k = K_range[np.argmin(sse_values)]
    max_sil_k = K_range[np.argmax(sil_scores)]
    
    print(f"\nPAPILDOMOS STATISTIKOS:")
    print(f"  Mažiausias SSE:         k = {min_sse_k} (SSE = {min(sse_values):.2f})")
    print(f"  Didžiausias siluetas:   k = {max_sil_k} (siluetas = {max(sil_scores):.4f})")
    
    # Rekomendacija
    methods_votes = [k_emp, k_elbow, k_sil]
    from collections import Counter
    votes = Counter(methods_votes)
    
    if len(votes) == 1:
        recommended_k = list(votes.keys())[0]
        consensus = "Pilnas sutarimas"
    else:
        recommended_k = votes.most_common(1)[0][0]
        consensus = f"Dalinis sutarimas ({votes[recommended_k]}/3 metodai)"
    
    print(f"\nREKOMENDACIJA:")
    print(f"  Rekomenduojamas k:      {recommended_k}")
    print(f"  Metodų sutarimas:       {consensus}")
    
    if abs(k_elbow - k_sil) <= 1:
        print(f"\nIŠVADOS:")
        print(f"  Elbow ir silueto metodai duoda panašius rezultatus")
        print(f"  Rekomenduojama naudoti k = {recommended_k}")
    else:
        print(f"\nIŠVADOS:")
        print(f"  Metodai duoda skirtingus rezultatus")
        print(f"  Siūloma išbandyti k = {k_sil}")
    
    print(f"\n  Grafikai išsaugoti 'outputs/' kataloge")
    print("="*60)
    
    # Išsaugome rezultatus - KONVERTUOJAME NUMPY TIPUS Į PYTHON TIPUS
    import json
    results = {
        'data_info': {
            'n_samples': int(X_shape[0]),  # numpy int -> python int
            'n_features': int(X_shape[1])  # numpy int -> python int
        },
        'methods': {
            'empirical': int(k_emp),       # numpy int -> python int
            'elbow': int(k_elbow),         # numpy int -> python int
            'silhouette': int(k_sil)       # numpy int -> python int
        },
        'recommendation': {
            'optimal_k': int(recommended_k),  # numpy int -> python int
            'consensus': consensus
        }
    }
    
    with open('outputs/optimal_k_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return recommended_k

def get_optimal_k():
    """
    Grąžina optimalų k iš išsaugotų rezultatų
    """
    try:
        import json
        with open('outputs/optimal_k_results.json', 'r') as f:
            results = json.load(f)
        return results['recommendation']['optimal_k']
    except:
        return None

def main():
    """
    Pagrindinė funkcija
    """
    print("OPTIMALAUS KLASTERIŲ SKAIČIAUS NUSTATYMAS")
    print("Metodai: Empirinis, Elbow, Vidutinio silueto")
    print("-" * 60)
    
    os.makedirs('outputs', exist_ok=True)
    
    # Duomenų paruošimas
    X, df, feature_names = load_and_prepare_data()
    
    # Empirinis metodas
    k_empirical = empirical_method(X)
    
    # Elbow metodas
    max_k = min(8, len(X) // 20)
    k_elbow, sse_values, K_range = elbow_method(X, max_k)
    
    # Silueto metodas
    k_silhouette, sil_scores, _ = silhouette_method(X, max_k)
    
    # Kombinuota analizė
    combined_analysis_plot(K_range, sse_values, sil_scores, k_empirical, k_elbow, k_silhouette)
    
    # Ataskaitos generavimas
    recommended_k = generate_report(k_empirical, k_elbow, k_silhouette, 
                                   sse_values, sil_scores, X.shape, K_range)
    
    print(f"\nANALIZĖ BAIGTA!")
    print(f"REKOMENDUOJAMAS k = {recommended_k}")
    print(f"Rezultatai išsaugoti 'outputs/' kataloge")
    
    return recommended_k

if __name__ == "__main__":
    optimal_k = main()