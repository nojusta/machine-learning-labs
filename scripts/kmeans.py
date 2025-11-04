import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import os
import json
import warnings
warnings.filterwarnings('ignore')

def load_optimal_k():
    """
    Užkrauna optimalų k iš opt_cluster.py rezultatų
    """
    try:
        with open('outputs/optimal_k_results.json', 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        print("=== UŽKRAUTI OPTIMALAUS K REZULTATAI ===")
        print(f"Empirinis metodas: k = {results['methods']['empirical']}")
        print(f"Elbow metodas: k = {results['methods']['elbow']}")
        print(f"Silueto metodas: k = {results['methods']['silhouette']}")
        print(f"Rekomenduojamas k: {results['recommendation']['optimal_k']}")
        print(f"Metodų sutarimas: {results['recommendation']['consensus']}")
        
        return results
    except FileNotFoundError:
        print("KLAIDA: Nerasti opt_cluster.py rezultatai!")
        print("Paleiskite 'python scripts/opt_cluster.py' pirmiausia.")
        return None
    except Exception as e:
        print(f"KLAIDA užkraunant rezultatus: {e}")
        return None

def load_and_prepare_data(use_all_features=False):
    """
    Duomenų užkrovimas - naudoja tą pačią logiką kaip opt_cluster.py
    """
    print("\n=== DUOMENŲ PARUOŠIMAS K-MEANS ===")
    
    if use_all_features:
        filename = "normalized_minmax_all.csv"
        print("Naudojami visi požymiai")
    else:
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
    
    # Išskiriame target kintamąjį jei egzistuoja
    target = None
    if 'NObeyesdad' in df.columns:
        target = df['NObeyesdad'].values
        print(f"Rastas target kintamasis (klasių skaičius: {len(np.unique(target))})")
    
    # Pasiimame tik skaitinius požymius (be target)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if 'NObeyesdad' in numeric_cols:
        numeric_cols = numeric_cols.drop('NObeyesdad')
    
    X = df[numeric_cols].values
    
    print(f"Duomenų forma: {X.shape}")
    print(f"Naudojami požymiai: {list(numeric_cols)}")
    
    # Standartizavimas (kaip opt_cluster.py)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, target, list(numeric_cols), df

def perform_kmeans(X, n_clusters, random_state=42):
    """
    Atlieka K-means klasterizaciją
    """
    print(f"\n=== K-MEANS KLASTERIZACIJA (k={n_clusters}) ===")
    
    # K-means modelis
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        max_iter=300
    )
    
    # Klasterizacija
    cluster_labels = kmeans.fit_predict(X)
    
    print(f"Klasterizacija baigta!")
    print(f"Klasterių centrai: {kmeans.cluster_centers_.shape}")
    print(f"Iteracijų skaičius: {kmeans.n_iter_}")
    print(f"SSE (inertia): {kmeans.inertia_:.2f}")
    
    return kmeans, cluster_labels

def evaluate_clustering(X, cluster_labels, target=None):
    """
    Vertina klasterizacijos kokybę
    """
    print(f"\n=== KLASTERIZACIJOS VERTINIMAS ===")
    
    # Vidiniai vertinimo kriterijai
    silhouette_avg = silhouette_score(X, cluster_labels)
    calinski_avg = calinski_harabasz_score(X, cluster_labels)
    davies_bouldin_avg = davies_bouldin_score(X, cluster_labels)
    
    print(f"Silhouette Score: {silhouette_avg:.4f}")
    print(f"Calinski-Harabasz Score: {calinski_avg:.2f}")
    print(f"Davies-Bouldin Score: {davies_bouldin_avg:.4f}")
    
    # Klasterių statistikos
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    print(f"\nKlasterių pasiskirstymas:")
    for label, count in zip(unique_labels, counts):
        percentage = (count / len(cluster_labels)) * 100
        print(f"  Klasteris {label}: {count} taškai ({percentage:.1f}%)")
    
    # Išorinis vertinimas (jei yra target)
    if target is not None:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        ari = adjusted_rand_score(target, cluster_labels)
        nmi = normalized_mutual_info_score(target, cluster_labels)
        
        print(f"\nIšorinis vertinimas (su tikrais labeliais):")
        print(f"Adjusted Rand Index: {ari:.4f}")
        print(f"Normalized Mutual Information: {nmi:.4f}")
    
    return {
        'silhouette': silhouette_avg,
        'calinski_harabasz': calinski_avg,
        'davies_bouldin': davies_bouldin_avg,
        'cluster_counts': dict(zip(unique_labels, counts))
    }

def visualize_clusters_2d(X, cluster_labels, kmeans, feature_names, target=None):
    """
    2D vizualizacija su PCA
    """
    print(f"\n=== 2D VIZUALIZACIJA ===")
    
    # PCA 2 komponentams
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Centrai PCA erdvėje
    centers_pca = pca.transform(kmeans.cluster_centers_)
    
    # Grafiko sukūrimas
    fig, axes = plt.subplots(1, 2 if target is not None else 1, figsize=(15, 6))
    if target is None:
        axes = [axes]
    
    # K-means rezultatai
    scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], 
                              c=cluster_labels, cmap='viridis', 
                              alpha=0.7, s=50)
    axes[0].scatter(centers_pca[:, 0], centers_pca[:, 1], 
                   c='red', marker='X', s=200, linewidths=2,
                   label='Centrai')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} dispersijos)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} dispersijos)')
    axes[0].set_title('K-means klasteriai (PCA)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0])
    
    # Tikri labeliai (jei yra)
    if target is not None:
        scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], 
                                  c=target, cmap='Set1', 
                                  alpha=0.7, s=50)
        axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} dispersijos)')
        axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} dispersijos)')
        axes[1].set_title('Tikri labeliai (PCA)')
        axes[1].grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=axes[1])
    
    plt.tight_layout()
    
    # Išsaugojimas
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/kmeans_clusters_2d.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"PCA paaiškinta dispersija: {sum(pca.explained_variance_ratio_):.2%}")

def cluster_analysis_report(kmeans, cluster_labels, X, feature_names, metrics):
    """
    Išsami klasterių analizės ataskaita
    """
    print(f"\n=== KLASTERIŲ ANALIZĖS ATASKAITA ===")
    
    n_clusters = len(kmeans.cluster_centers_)
    
    # Klasterių centrai
    centers_df = pd.DataFrame(
        kmeans.cluster_centers_, 
        columns=feature_names,
        index=[f'Klasteris_{i}' for i in range(n_clusters)]
    )
    
    print(f"\nKlasterių centrai (standartizuoti duomenys):")
    print(centers_df.round(3))
    
    # Klasterių charakteristikos
    print(f"\nKlasterių charakteristikos:")
    for i in range(n_clusters):
        cluster_mask = cluster_labels == i
        cluster_size = np.sum(cluster_mask)
        cluster_data = X[cluster_mask]
        
        print(f"\n--- Klasteris {i} ---")
        print(f"Dydis: {cluster_size} taškai")
        print(f"Vidutinis atstumas iki centro: {np.mean(np.linalg.norm(cluster_data - kmeans.cluster_centers_[i], axis=1)):.3f}")
        
        # Svarbiausieji požymiai (didžiausi centrai)
        center_values = kmeans.cluster_centers_[i]
        important_features = np.argsort(np.abs(center_values))[-3:][::-1]
        print(f"Svarbiausieji požymiai:")
        for idx in important_features:
            print(f"  {feature_names[idx]}: {center_values[idx]:.3f}")
    
    # Išsaugojimas
    results_summary = {
        'n_clusters': int(n_clusters),
        'total_samples': int(len(cluster_labels)),
        'sse': float(kmeans.inertia_),
        'iterations': int(kmeans.n_iter_),
        'metrics': metrics,
        'cluster_centers': centers_df.to_dict(),
        'cluster_sizes': {int(k): int(v) for k, v in metrics['cluster_counts'].items()}
    }
    
    with open('outputs/kmeans_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nAtalkaita išsaugota: outputs/kmeans_results.json")

def compare_different_k_values(X, optimal_results, max_k=8):
    """
    Palygina K-means su skirtingais k (aplink optimalų)
    """
    print(f"\n=== K-MEANS PALYGINIMAS SU SKIRTINGAIS K ===")
    
    recommended_k = optimal_results['recommendation']['optimal_k']
    k_values = range(max(2, recommended_k-2), min(max_k+1, recommended_k+3))
    
    results = []
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        if len(set(labels)) > 1:  # Tik jei yra keli klasteriai
            sil_score = silhouette_score(X, labels)
            cal_score = calinski_harabasz_score(X, labels)
            db_score = davies_bouldin_score(X, labels)
        else:
            sil_score = cal_score = db_score = 0
        
        results.append({
            'k': k,
            'silhouette': sil_score,
            'calinski_harabasz': cal_score,
            'davies_bouldin': db_score,
            'sse': kmeans.inertia_
        })
        
        marker = " ← REKOMENDUOJAMAS" if k == recommended_k else ""
        print(f"k={k}: Silhouette={sil_score:.3f}, SSE={kmeans.inertia_:.1f}{marker}")
    
    # Vizualizacija
    results_df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0,0].plot(results_df['k'], results_df['silhouette'], 'o-', color='green')
    axes[0,0].axvline(x=recommended_k, color='red', linestyle='--', alpha=0.7)
    axes[0,0].set_title('Silhouette Score')
    axes[0,0].set_xlabel('k')
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].plot(results_df['k'], results_df['sse'], 'o-', color='blue')
    axes[0,1].axvline(x=recommended_k, color='red', linestyle='--', alpha=0.7)
    axes[0,1].set_title('SSE (Inertia)')
    axes[0,1].set_xlabel('k')
    axes[0,1].grid(True, alpha=0.3)
    
    axes[1,0].plot(results_df['k'], results_df['calinski_harabasz'], 'o-', color='orange')
    axes[1,0].axvline(x=recommended_k, color='red', linestyle='--', alpha=0.7)
    axes[1,0].set_title('Calinski-Harabasz Score')
    axes[1,0].set_xlabel('k')
    axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].plot(results_df['k'], results_df['davies_bouldin'], 'o-', color='purple')
    axes[1,1].axvline(x=recommended_k, color='red', linestyle='--', alpha=0.7)
    axes[1,1].set_title('Davies-Bouldin Score')
    axes[1,1].set_xlabel('k')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.suptitle(f'K-means su skirtingais k (rekomenduojamas k={recommended_k})')
    plt.tight_layout()
    plt.savefig('outputs/kmeans_k_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

def main(use_all_features=False):
    """
    Pagrindinė K-means analizės funkcija
    """
    print("K-MEANS KLASTERIZACIJOS ANALIZĖ")
    print("Naudoja opt_cluster.py rezultatus")
    print("-" * 60)
    
    os.makedirs('outputs', exist_ok=True)
    
    # 1. Užkrauname optimalaus k rezultatus
    optimal_results = load_optimal_k()
    if optimal_results is None:
        return None
    
    recommended_k = optimal_results['recommendation']['optimal_k']
    
    # 2. Užkrauname duomenis
    X, target, feature_names, df = load_and_prepare_data(use_all_features)
    
    # 3. Atliekame K-means su rekomenduojamu k
    kmeans, cluster_labels = perform_kmeans(X, recommended_k)
    
    # 4. Vertiname rezultatus
    metrics = evaluate_clustering(X, cluster_labels, target)
    
    # 5. Vizualizuojame
    visualize_clusters_2d(X, cluster_labels, kmeans, feature_names, target)
    
    # 6. Detali ataskaita
    cluster_analysis_report(kmeans, cluster_labels, X, feature_names, metrics)
    
    # 7. Palyginame su kitais k
    comparison_results = compare_different_k_values(X, optimal_results)
    
    print(f"\nK-MEANS ANALIZĖ BAIGTA!")
    print(f"Naudotas k = {recommended_k}")
    print(f"Silhouette Score = {metrics['silhouette']:.4f}")
    print(f"Rezultatai išsaugoti 'outputs/' kataloge")
    
    return {
        'kmeans': kmeans,
        'labels': cluster_labels,
        'metrics': metrics,
        'optimal_k': recommended_k
    }

if __name__ == "__main__":
    # Naudojame tą patį požymių rinkinį kaip opt_cluster.py
    results = main(use_all_features=False)