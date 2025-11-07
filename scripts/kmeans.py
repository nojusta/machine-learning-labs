import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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

def load_full_dataset():
    """
    1. Pilna duomenų aibė (visi požymiai be NObeyesdad)
    """
    print("\n=== 1. PILNOS DUOMENŲ AIBĖS PARUOŠIMAS ===")
    
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "data", "normalized_minmax_all.csv"),
        os.path.join(".", "data", "normalized_minmax_all.csv"),
        "../data/normalized_minmax_all.csv"
    ]
    
    df = None
    for path in candidates:
        if os.path.isfile(path):
            df = pd.read_csv(path)
            print(f"Užkrautas failas: {path}")
            break
    
    if df is None:
        raise FileNotFoundError("Nerastas normalized_minmax_all.csv failas")
    
    # Target kintamasis
    target = None
    if 'NObeyesdad' in df.columns:
        target = df['NObeyesdad'].values
        print(f"Rastas target kintamasis (klasių skaičius: {len(np.unique(target))})")
    
    # Visi požymiai be target
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if 'NObeyesdad' in numeric_cols:
        numeric_cols = numeric_cols.drop('NObeyesdad')
    
    X = df[numeric_cols].values
    feature_names = list(numeric_cols)
    
    print(f"Pilnos aibės forma: {X.shape}")
    print(f"Visi požymiai ({len(feature_names)}): {feature_names}")
    
    # Standartizavimas
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, target, feature_names, "Pilna duomenų aibė"

def load_chi2_dataset():
    """
    2. Chi² atrinkti požymiai: Gender, FCVC, SMOKE, CALC, NCP, CH2O
    """
    print("\n=== 2. Chi² ATRINKTŲ POŽYMIŲ PARUOŠIMAS ===")
    
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "data", "normalized_minmax.csv"),
        os.path.join(".", "data", "normalized_minmax.csv"),
        "../data/normalized_minmax.csv"
    ]
    
    df = None
    for path in candidates:
        if os.path.isfile(path):
            df = pd.read_csv(path)
            print(f"Užkrautas failas: {path}")
            break
    
    if df is None:
        raise FileNotFoundError("Nerastas normalized_minmax.csv failas")
    
    # Target kintamasis
    target = None
    if 'NObeyesdad' in df.columns:
        target = df['NObeyesdad'].values
    
    # Chi² atrinkti požymiai
    chi2_features = ['Gender', 'FCVC', 'SMOKE', 'CALC', 'NCP', 'CH2O']
    available_features = [f for f in chi2_features if f in df.columns]
    
    X = df[available_features].values
    
    print(f"Chi² aibės forma: {X.shape}")
    print(f"Chi² požymiai ({len(available_features)}): {available_features}")
    
    # Standartizavimas
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, target, available_features, "Chi² atrinkti požymiai"

def create_tsne_dataset():
    """
    3. t-SNE sumažintos dimensijos rinkinys (Chi² → 2D)
    """
    print("\n=== 3. t-SNE SUMAŽINTŲ DIMENSIJŲ PARUOŠIMAS ===")
    
    # Pirmiausia gauname Chi² duomenis
    X_chi2, target, chi2_features, _ = load_chi2_dataset()
    
    print("Taikomas t-SNE dimensijų mažinimas...")
    
    # t-SNE parametrai
    perplexity = min(30, X_chi2.shape[0] // 4) if X_chi2.shape[0] > 30 else 5
    
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity,
        max_iter=1000,
        learning_rate='auto'
    )
    
    X_tsne = tsne.fit_transform(X_chi2)
    
    feature_names = ['t-SNE_1', 't-SNE_2']
    
    print(f"t-SNE aibės forma: {X_tsne.shape}")
    print(f"t-SNE požymiai: {feature_names}")
    print(f"Naudotas perplexity: {perplexity}")
    
    return X_tsne, target, feature_names, "t-SNE sumažintos dimensijos"

def perform_kmeans_analysis(X, target, feature_names, dataset_name, recommended_k):
    """
    Atlieka pilną K-means analizę vienam duomenų rinkiniui
    """
    print(f"\n{'='*20} {dataset_name.upper()} {'='*20}")
    
    # K-means klasterizacija
    kmeans = KMeans(
        n_clusters=recommended_k,
        random_state=42,
        n_init=10,
        max_iter=300
    )
    
    cluster_labels = kmeans.fit_predict(X)
    
    print(f"K-means baigtas (k={recommended_k})")
    print(f"SSE (inertia): {kmeans.inertia_:.2f}")
    print(f"Iteracijų skaičius: {kmeans.n_iter_}")
    
    # Vertinimo metrikos
    silhouette_avg = silhouette_score(X, cluster_labels)
    calinski_avg = calinski_harabasz_score(X, cluster_labels)
    davies_bouldin_avg = davies_bouldin_score(X, cluster_labels)
    
    print(f"\nVertinimo metrikos:")
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
    external_metrics = {}
    if target is not None:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        ari = adjusted_rand_score(target, cluster_labels)
        nmi = normalized_mutual_info_score(target, cluster_labels)
        
        external_metrics = {'ari': ari, 'nmi': nmi}
        print(f"\nIšorinis vertinimas:")
        print(f"Adjusted Rand Index: {ari:.4f}")
        print(f"Normalized Mutual Information: {nmi:.4f}")
    
    metrics = {
        'silhouette': silhouette_avg,
        'calinski_harabasz': calinski_avg,
        'davies_bouldin': davies_bouldin_avg,
        'cluster_counts': dict(zip(unique_labels, counts)),
        'external_metrics': external_metrics
    }
    
    return kmeans, cluster_labels, metrics

def visualize_all_datasets(datasets_results, recommended_k):
    """
    Vizualizuoja visus tris duomenų rinkinius
    """
    print(f"\n=== VISŲ DUOMENŲ RINKINIŲ VIZUALIZACIJA ===")
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    for i, (dataset_name, results) in enumerate(datasets_results.items()):
        X = results['data']
        cluster_labels = results['cluster_labels']
        target = results['target']
        feature_names = results['feature_names']
        
        # Jei duomenys nėra 2D, taikome PCA
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_plot = pca.fit_transform(X)
            x_label = f'PC1 ({pca.explained_variance_ratio_[0]:.1%})'
            y_label = f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'
        else:
            X_plot = X
            x_label = feature_names[0] if len(feature_names) > 0 else 'Dim 1'
            y_label = feature_names[1] if len(feature_names) > 1 else 'Dim 2'
        
        # 1. K-means klasteriai
        scatter1 = axes[i, 0].scatter(X_plot[:, 0], X_plot[:, 1], 
                                     c=cluster_labels, cmap='viridis', 
                                     alpha=0.7, s=30)
        axes[i, 0].set_title(f'{dataset_name}\nK-means klasteriai')
        axes[i, 0].set_xlabel(x_label)
        axes[i, 0].set_ylabel(y_label)
        axes[i, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=axes[i, 0])
        
        # 2. Tikri labeliai (jei yra)
        if target is not None:
            scatter2 = axes[i, 1].scatter(X_plot[:, 0], X_plot[:, 1], 
                                         c=target, cmap='Set1', 
                                         alpha=0.7, s=30)
            axes[i, 1].set_title(f'{dataset_name}\nTikri labeliai')
            axes[i, 1].set_xlabel(x_label)
            axes[i, 1].set_ylabel(y_label)
            axes[i, 1].grid(True, alpha=0.3)
            plt.colorbar(scatter2, ax=axes[i, 1])
        else:
            axes[i, 1].axis('off')
            axes[i, 1].text(0.5, 0.5, 'Nėra target\nkintamojo', 
                           ha='center', va='center', transform=axes[i, 1].transAxes)
        
        # 3. Klasterių statistikos
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        axes[i, 2].bar(unique_labels, counts, color='lightblue', alpha=0.7)
        axes[i, 2].set_title(f'{dataset_name}\nKlasterių dydžiai')
        axes[i, 2].set_xlabel('Klasteris')
        axes[i, 2].set_ylabel('Taškų skaičius')
        axes[i, 2].grid(True, alpha=0.3)
    
    plt.suptitle(f'K-means analizė su trimis duomenų rinkiniais (k={recommended_k})', 
                 fontsize=16, y=0.98)
    plt.tight_layout()
    
    # Išsaugojimas
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/kmeans_three_datasets_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_comparison_report(datasets_results, recommended_k):
    """
    Generuoja išsamų palyginimo raportą
    """
    print(f"\n{'='*60}")
    print("           K-MEANS ANALIZĖS PALYGINIMO ATASKAITA")
    print(f"{'='*60}")
    
    print(f"\nNAUDOTAS KLASTERIŲ SKAIČIUS: k = {recommended_k}")
    print(f"(Pagal opt_cluster.py rekomendaciją)")
    
    # Metrikos palyginimas
    print(f"\n{'DUOMENŲ RINKINYS':<25} {'SILHOUETTE':<12} {'CALINSKI-H':<12} {'DAVIES-B':<12} {'SSE':<10}")
    print("-" * 75)
    
    for dataset_name, results in datasets_results.items():
        metrics = results['metrics']
        sse = results['kmeans'].inertia_
        
        print(f"{dataset_name:<25} {metrics['silhouette']:<12.4f} "
              f"{metrics['calinski_harabasz']:<12.2f} {metrics['davies_bouldin']:<12.4f} "
              f"{sse:<10.1f}")
    
    # Išorinis vertinimas
    print(f"\nIŠORINIS VERTINIMAS (su tikrais labeliais):")
    print(f"{'DUOMENŲ RINKINYS':<25} {'ARI':<10} {'NMI':<10}")
    print("-" * 50)
    
    for dataset_name, results in datasets_results.items():
        external = results['metrics']['external_metrics']
        if external:
            print(f"{dataset_name:<25} {external['ari']:<10.4f} {external['nmi']:<10.4f}")
        else:
            print(f"{dataset_name:<25} {'N/A':<10} {'N/A':<10}")
    
    # Klasterių pasiskirstymas
    print(f"\nKLASTERIŲ PASISKIRSTYMAS:")
    for dataset_name, results in datasets_results.items():
        counts = results['metrics']['cluster_counts']
        print(f"\n{dataset_name}:")
        for cluster, count in counts.items():
            percentage = (count / sum(counts.values())) * 100
            print(f"  Klasteris {cluster}: {count} taškai ({percentage:.1f}%)")
    
    # Rekomendacijos
    print(f"\n{'='*60}")
    print("REKOMENDACIJOS IR IŠVADOS:")
    print(f"{'='*60}")
    
    # Geriausias Silhouette
    best_silhouette = max(datasets_results.items(), 
                         key=lambda x: x[1]['metrics']['silhouette'])
    
    print(f"\n1. GERIAUSIAS SILHOUETTE SCORE:")
    print(f"   {best_silhouette[0]} - {best_silhouette[1]['metrics']['silhouette']:.4f}")
    
    # Geriausias išorinis vertinimas
    best_external = None
    best_ari = -1
    for name, results in datasets_results.items():
        external = results['metrics']['external_metrics']
        if external and external['ari'] > best_ari:
            best_ari = external['ari']
            best_external = (name, external)
    
    if best_external:
        print(f"\n2. GERIAUSIAS IŠORINIS VERTINIMAS (ARI):")
        print(f"   {best_external[0]} - {best_external[1]['ari']:.4f}")
    
    print(f"\n3. DUOMENŲ RINKINIŲ CHARAKTERISTIKOS:")
    for dataset_name, results in datasets_results.items():
        shape = results['data'].shape
        print(f"   {dataset_name}: {shape[0]} taškai, {shape[1]} požymiai")
    
    print(f"\n4. REKOMENDACIJOS:")
    print(f"   - Dimensijų mažinimas (t-SNE) gali pagerinti vizualizaciją")
    print(f"   - Chi² požymių atrinkimas sumažina triukšmą")
    print(f"   - Pilna duomenų aibė išlaiko maksimalų informacijos kiekį")
    
    print(f"\nANALIZĖ BAIGTA! Rezultatai išsaugoti 'outputs/' kataloge.")
    
    # JSON išsaugojimas
    comparison_data = {
        'recommended_k': recommended_k,
        'datasets': {}
    }
    
    for dataset_name, results in datasets_results.items():
        comparison_data['datasets'][dataset_name] = {
            'shape': results['data'].shape,
            'features': results['feature_names'],
            'metrics': {
                'silhouette': float(results['metrics']['silhouette']),
                'calinski_harabasz': float(results['metrics']['calinski_harabasz']),
                'davies_bouldin': float(results['metrics']['davies_bouldin']),
                'sse': float(results['kmeans'].inertia_)
            },
            'cluster_sizes': results['metrics']['cluster_counts'],
            'external_metrics': results['metrics']['external_metrics']
        }
    
    with open('outputs/kmeans_three_datasets_comparison.json', 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)

def main():
    """
    Pagrindinė funkcija - analizuoja tris duomenų rinkinius
    """
    print("K-MEANS KLASTERIZACIJOS ANALIZĖ")
    print("Trys duomenų rinkiniai: Pilnas, Chi², t-SNE")
    print("="*60)
    
    os.makedirs('outputs', exist_ok=True)
    
    # 1. Užkrauname optimalaus k rezultatus
    optimal_results = load_optimal_k()
    if optimal_results is None:
        return None
    
    recommended_k = optimal_results['recommendation']['optimal_k']
    
    # 2. Paruošiame tris duomenų rinkinius
    datasets = []
    
    try:
        # 2.1 Pilna duomenų aibė
        X_full, target_full, features_full, name_full = load_full_dataset()
        datasets.append((X_full, target_full, features_full, name_full))
        
        # 2.2 Chi² atrinkti požymiai
        X_chi2, target_chi2, features_chi2, name_chi2 = load_chi2_dataset()
        datasets.append((X_chi2, target_chi2, features_chi2, name_chi2))
        
        # 2.3 t-SNE sumažintos dimensijos
        X_tsne, target_tsne, features_tsne, name_tsne = create_tsne_dataset()
        datasets.append((X_tsne, target_tsne, features_tsne, name_tsne))
        
    except Exception as e:
        print(f"KLAIDA ruošiant duomenis: {e}")
        return None
    
    # 3. Atliekame K-means analizę visiems rinkiniams
    datasets_results = {}
    
    for X, target, feature_names, dataset_name in datasets:
        kmeans, cluster_labels, metrics = perform_kmeans_analysis(
            X, target, feature_names, dataset_name, recommended_k
        )
        
        datasets_results[dataset_name] = {
            'data': X,
            'target': target,
            'feature_names': feature_names,
            'kmeans': kmeans,
            'cluster_labels': cluster_labels,
            'metrics': metrics
        }
    
    # 4. Vizualizacija
    visualize_all_datasets(datasets_results, recommended_k)
    
    # 5. Palyginimo ataskaita
    generate_comparison_report(datasets_results, recommended_k)
    
    return datasets_results

if __name__ == "__main__":
    results = main()