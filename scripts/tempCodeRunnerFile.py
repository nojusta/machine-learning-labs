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
from collections import Counter
warnings.filterwarnings('ignore')

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

def find_optimal_k(X, max_k=10):
    """
    Apskaičiuoja optimalų klasterių skaičių naudojant:
    - Empirinį metodą
    - Elbow (SSE)
    - Silhouette Score
    """
    print("\n=== OPTIMALAUS K APSKAIČIAVIMAS ===")
    
    n = X.shape[0]
    k_emp = int(np.sqrt(n / 2))
    print(f"Empirinis metodas → k = {k_emp}")
    
    # Ribojame max_k pagal duomenų kiekį
    max_k = min(max_k, n // 5)  # Ne daugiau nei n/5
    if max_k < 3:
        max_k = 3
    
    sse = []
    sil_scores = []
    K_range = range(2, max_k + 1)
    
    print(f"Testuojami k nuo 2 iki {max_k}...")
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=200)
        labels = kmeans.fit_predict(X)
        sse.append(kmeans.inertia_)
        sil = silhouette_score(X, labels)
        sil_scores.append(sil)
        print(f"k={k}: SSE={kmeans.inertia_:.2f}, Silhouette={sil:.4f}")
    
    # Elbow metodas (alkūnės taškas)
    deltas = np.diff(sse)
    second_diff = np.diff(deltas)
    k_elbow = int(K_range[np.argmin(second_diff) + 1]) if len(second_diff) > 0 else 3
    print(f"Elbow metodas → k = {k_elbow}")
    
    # Silhouette metodas
    k_sil = int(K_range[np.argmax(sil_scores)])
    print(f"Silueto metodas → k = {k_sil}")
    
    # Daugumos balsavimas
    votes = [k_emp, k_elbow, k_sil]
    recommended_k = Counter(votes).most_common(1)[0][0]
    
    # Saugumas - jei rekomenduojamas k per didelis
    if recommended_k > max_k:
        recommended_k = max_k
    elif recommended_k < 2:
        recommended_k = 2
    
    print(f"\n=== METODŲ REZULTATAI ===")
    print(f"Empirinis: {k_emp}")
    print(f"Elbow: {k_elbow}")  
    print(f"Silhouette: {k_sil}")
    print(f"Balsai: {votes}")
    print(f"REKOMENDUOJAMAS K = {recommended_k} (pagal daugumos balsą)")
    
    # Vizualizacijos
    os.makedirs('outputs', exist_ok=True)
    
    # Elbow grafikas
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(K_range, sse, marker='o', linewidth=2, markersize=8, color='blue')
    plt.axvline(x=k_elbow, color='red', linestyle='--', alpha=0.7, 
                label=f'Elbow k={k_elbow}')
    plt.xlabel("Klasterių skaičius (k)")
    plt.ylabel("SSE (Inertia)")
    plt.title("Elbow metodas")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Silhouette grafikas
    plt.subplot(1, 2, 2)
    plt.plot(K_range, sil_scores, marker='o', linewidth=2, markersize=8, color='green')
    plt.axvline(x=k_sil, color='red', linestyle='--', alpha=0.7,
                label=f'Max Silhouette k={k_sil}')
    plt.xlabel("Klasterių skaičius (k)")
    plt.ylabel("Vidutinis Silhouette Score")
    plt.title("Silhouette metodas")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.suptitle(f'Optimalaus k metodai (Rekomenduojama: k={recommended_k})', fontsize=14)
    plt.tight_layout()
    plt.savefig("outputs/optimal_k_methods.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Sukurtas grafikas: outputs/optimal_k_methods.png")

    return {
        "empirical": k_emp,
        "elbow": k_elbow,
        "silhouette": k_sil,
        "recommended": recommended_k,
        "sse_values": sse,
        "silhouette_values": sil_scores,
        "k_range": list(K_range)
    }

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

def visualize_individual_datasets(datasets_results, recommended_k):
    """
    Sukuria atskirus PNG failus kiekvienam duomenų rinkiniui
    """
    print(f"\n=== INDIVIDUALIŲ DUOMENŲ RINKINIŲ VIZUALIZACIJA ===")
    
    os.makedirs('outputs', exist_ok=True)
    
    for dataset_name, results in datasets_results.items():
        X = results['data']
        cluster_labels = results['cluster_labels']
        target = results['target']
        feature_names = results['feature_names']
        kmeans = results['kmeans']
        metrics = results['metrics']
        
        # Safe filename
        safe_name = dataset_name.replace(' ', '_').replace('²', '2')
        
        # 1. Klasterių vizualizacija
        create_cluster_visualization(X, cluster_labels, target, feature_names, 
                                   dataset_name, safe_name, recommended_k, kmeans)
        
        # 2. Metrikos grafikai
        create_metrics_visualization(metrics, dataset_name, safe_name, recommended_k)
        
        # 3. Klasterių charakteristikos
        create_cluster_characteristics(X, cluster_labels, kmeans, feature_names, 
                                     dataset_name, safe_name)
        
        print(f"Sukurti grafikai duomenų rinkiniui: {dataset_name}")

def create_cluster_visualization(X, cluster_labels, target, feature_names, 
                               dataset_name, safe_name, recommended_k, kmeans):
    """
    Sukuria klasterių vizualizacijos grafiką
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Jei duomenys nėra 2D, taikome PCA
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_plot = pca.fit_transform(X)
        x_label = f'PC1 ({pca.explained_variance_ratio_[0]:.1%})'
        y_label = f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'
        
        # Klasterių centrai PCA erdvėje
        if hasattr(kmeans, 'cluster_centers_'):
            centers_pca = pca.transform(kmeans.cluster_centers_)
        else:
            centers_pca = None
    else:
        X_plot = X
        x_label = feature_names[0] if len(feature_names) > 0 else 'Dim 1'
        y_label = feature_names[1] if len(feature_names) > 1 else 'Dim 2'
        centers_pca = kmeans.cluster_centers_ if hasattr(kmeans, 'cluster_centers_') else None
    
    # 1. K-means klasteriai
    scatter1 = axes[0].scatter(X_plot[:, 0], X_plot[:, 1], 
                              c=cluster_labels, cmap='viridis', 
                              alpha=0.7, s=50)
    
    # Centrai
    if centers_pca is not None:
        axes[0].scatter(centers_pca[:, 0], centers_pca[:, 1], 
                       c='red', marker='X', s=200, linewidths=2,
                       label='Centrai', edgecolors='black')
        axes[0].legend()
    
    axes[0].set_title(f'K-means klasteriai (k={recommended_k})')
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(y_label)
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0])
    
    # 2. Tikri labeliai (jei yra)
    if target is not None:
        scatter2 = axes[1].scatter(X_plot[:, 0], X_plot[:, 1], 
                                  c=target, cmap='Set1', 
                                  alpha=0.7, s=50)
        axes[1].set_title('Tikri labeliai')
        axes[1].set_xlabel(x_label)
        axes[1].set_ylabel(y_label)
        axes[1].grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=axes[1])
    else:
        axes[1].axis('off')
        axes[1].text(0.5, 0.5, 'Nėra tikrų\nlabelių', 
                    ha='center', va='center', transform=axes[1].transAxes,
                    fontsize=14, bbox=dict(boxstyle="round", facecolor='lightgray'))
    
    # 3. Klasterių dydžiai
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    bars = axes[2].bar(unique_labels, counts, color=colors, alpha=0.7, edgecolor='black')
    axes[2].set_title('Klasterių dydžiai')
    axes[2].set_xlabel('Klasteris')
    axes[2].set_ylabel('Taškų skaičius')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    # Pridėti procentus ant stulpelių
    total_points = len(cluster_labels)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        percentage = (count / total_points) * 100
        axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{percentage:.1f}%', ha='center', va='bottom')
    
    plt.suptitle(f'{dataset_name} - Klasterių analizė', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Išsaugojimas
    plt.savefig(f'outputs/{safe_name}_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_metrics_visualization(metrics, dataset_name, safe_name, recommended_k):
    """
    Sukuria metrikos vizualizacijos grafiką
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Metrikos reikšmės
    metric_names = ['Silhouette\nScore', 'Calinski-Harabasz\nScore', 
                   'Davies-Bouldin\nScore']
    metric_values = [metrics['silhouette'], metrics['calinski_harabasz'], 
                    metrics['davies_bouldin']]
    
    # 1. Metrikos stulpeliai
    colors = ['green', 'blue', 'orange']
    bars = axes[0, 0].bar(range(3), metric_values, color=colors, alpha=0.7)
    axes[0, 0].set_title('Klasterizacijos metrikos')
    axes[0, 0].set_xticks(range(3))
    axes[0, 0].set_xticklabels(metric_names)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Reikšmės ant stulpelių
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.4f}', ha='center', va='bottom')
    
    # 2. Klasterių pasiskirstymo pyraginis grafikas
    cluster_counts = metrics['cluster_counts']
    labels = [f'Klasteris {k}' for k in cluster_counts.keys()]
    sizes = list(cluster_counts.values())
    colors_pie = plt.cm.viridis(np.linspace(0, 1, len(sizes)))
    
    wedges, texts, autotexts = axes[0, 1].pie(sizes, labels=labels, autopct='%1.1f%%', 
                                             colors=colors_pie, startangle=90)
    axes[0, 1].set_title('Klasterių pasiskirstymas')
    
    # 3. Išorinis vertinimas (jei yra)
    external = metrics['external_metrics']
    if external:
        ext_names = ['ARI', 'NMI']
        ext_values = [external['ari'], external['nmi']]
        bars_ext = axes[1, 0].bar(ext_names, ext_values, color=['purple', 'brown'], alpha=0.7)
        axes[1, 0].set_title('Išorinis vertinimas')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        axes[1, 0].set_ylim(0, 1)
        
        for bar, value in zip(bars_ext, ext_values):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.4f}', ha='center', va='bottom')
    else:
        axes[1, 0].axis('off')
        axes[1, 0].text(0.5, 0.5, 'Nėra išorinio\nvertinimo duomenų', 
                       ha='center', va='center', transform=axes[1, 0].transAxes,
                       fontsize=12, bbox=dict(boxstyle="round", facecolor='lightgray'))
    
    # 4. Metrikos interpretacija
    axes[1, 1].axis('off')
    
    # Metrikos interpretacijos tekstas
    cluster_counts_sum = sum(cluster_counts.values()) if cluster_counts else 0
    interpretation = f"""Metrikos interpretacija:

Silhouette Score: {metrics['silhouette']:.4f}
{'Puikus' if metrics['silhouette'] > 0.7 else 'Geras' if metrics['silhouette'] > 0.5 else 'Vidutinis' if metrics['silhouette'] > 0.3 else 'Prastas'} klasterizavimas

Calinski-Harabasz: {metrics['calinski_harabasz']:.2f}
{'Aukšta' if metrics['calinski_harabasz'] > 100 else 'Vidutinė'} klasterių kokybė

Davies-Bouldin: {metrics['davies_bouldin']:.4f}
{'Gerai atskirti' if metrics['davies_bouldin'] < 1.0 else 'Vidutiniškai atskirti'} klasteriai

Klasterių skaičius: {recommended_k}
Duomenų taškų: {cluster_counts_sum}
"""
    
    axes[1, 1].text(0.05, 0.95, interpretation, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    
    plt.suptitle(f'{dataset_name} - Vertinimo metrikos', fontsize=16)
    plt.tight_layout()
    
    # Išsaugojimas
    plt.savefig(f'outputs/{safe_name}_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_cluster_characteristics(X, cluster_labels, kmeans, feature_names, 
                                 dataset_name, safe_name):
    """
    Sukuria klasterių charakteristikų vizualizaciją
    """
    if not hasattr(kmeans, 'cluster_centers_'):
        print(f"Praleista {dataset_name} charakteristikos - nėra centrų")
        return
        
    n_clusters = len(kmeans.cluster_centers_)
    n_features = min(len(feature_names), 10)  # Apribojame iki 10 požymių
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 1. Klasterių centrai heatmap
    centers_subset = kmeans.cluster_centers_[:, :n_features]
    features_subset = feature_names[:n_features]
    
    im = axes[0].imshow(centers_subset, cmap='RdBu_r', aspect='auto')
    axes[0].set_title('Klasterių centrai (standartizuoti duomenys)')
    axes[0].set_xlabel('Požymiai')
    axes[0].set_ylabel('Klasteriai')
    axes[0].set_yticks(range(n_clusters))
    axes[0].set_yticklabels([f'Klasteris {i}' for i in range(n_clusters)])
    axes[0].set_xticks(range(len(features_subset)))
    axes[0].set_xticklabels(features_subset, rotation=45, ha='right')
    
    # Spalvų juosta
    cbar = plt.colorbar(im, ax=axes[0])
    cbar.set_label('Standartizuota reikšmė')
    
    # Reikšmių tekstas
    for i in range(n_clusters):
        for j in range(len(features_subset)):
            text = axes[0].text(j, i, f'{centers_subset[i, j]:.2f}',
                               ha="center", va="center", 
                               color="white" if abs(centers_subset[i, j]) > 1 else "black")
    
    # 2. Top požymiai kiekvienam klasteriui
    axes[1].axis('off')
    
    characteristics_text = f"Svarbiausieji požymiai kiekvienam klasteriui:\n\n"
    
    for i in range(n_clusters):
        center = kmeans.cluster_centers_[i]
        # Top 3 teigiami ir neigiami požymiai
        top_positive = np.argsort(center)[-3:][::-1]
        top_negative = np.argsort(center)[:3]
        
        cluster_size = sum(cluster_labels == i)
        characteristics_text += f"Klasteris {i} ({cluster_size} taškai):\n"
        characteristics_text += "  Aukščiausi požymiai:\n"
        
        for idx in top_positive:
            if idx < len(feature_names):
                characteristics_text += f"    {feature_names[idx]}: {center[idx]:.3f}\n"
        
        characteristics_text += "  Žemiausi požymiai:\n"
        for idx in top_negative:
            if idx < len(feature_names):
                characteristics_text += f"    {feature_names[idx]}: {center[idx]:.3f}\n"
        characteristics_text += "\n"
    
    axes[1].text(0.05, 0.95, characteristics_text, transform=axes[1].transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle(f'{dataset_name} - Klasterių charakteristikos', fontsize=16)
    plt.tight_layout()
    
    # Išsaugojimas
    plt.savefig(f'outputs/{safe_name}_characteristics.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_comparison(datasets_results, recommended_k):
    """
    Sukuria bendrą palyginimo grafiką
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    dataset_names = list(datasets_results.keys())
    
    # 1. Silhouette Score palyginimas
    silhouette_scores = [results['metrics']['silhouette'] for results in datasets_results.values()]
    colors = ['skyblue', 'lightgreen', 'lightcoral']
    
    bars1 = axes[0, 0].bar(range(len(dataset_names)), silhouette_scores, 
                          color=colors[:len(dataset_names)], alpha=0.7)
    axes[0, 0].set_title('Silhouette Score palyginimas')
    axes[0, 0].set_xlabel('Duomenų rinkinys')
    axes[0, 0].set_ylabel('Silhouette Score')
    axes[0, 0].set_xticks(range(len(dataset_names)))
    axes[0, 0].set_xticklabels(dataset_names, rotation=15, ha='right')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    for bar, score in zip(bars1, silhouette_scores):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{score:.4f}', ha='center', va='bottom')
    
    # 2. Calinski-Harabasz Score palyginimas
    calinski_scores = [results['metrics']['calinski_harabasz'] for results in datasets_results.values()]
    
    bars2 = axes[0, 1].bar(range(len(dataset_names)), calinski_scores, 
                          color=colors[:len(dataset_names)], alpha=0.7)
    axes[0, 1].set_title('Calinski-Harabasz Score palyginimas')
    axes[0, 1].set_xlabel('Duomenų rinkinys')
    axes[0, 1].set_ylabel('Calinski-Harabasz Score')
    axes[0, 1].set_xticks(range(len(dataset_names)))
    axes[0, 1].set_xticklabels(dataset_names, rotation=15, ha='right')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    for bar, score in zip(bars2, calinski_scores):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{score:.1f}', ha='center', va='bottom')
    
    # 3. Davies-Bouldin Score palyginimas (mažesnis = geriau)
    davies_scores = [results['metrics']['davies_bouldin'] for results in datasets_results.values()]
    
    bars3 = axes[1, 0].bar(range(len(dataset_names)), davies_scores, 
                          color=colors[:len(dataset_names)], alpha=0.7)
    axes[1, 0].set_title('Davies-Bouldin Score palyginimas')
    axes[1, 0].set_xlabel('Duomenų rinkinys')
    axes[1, 0].set_ylabel('Davies-Bouldin Score')
    axes[1, 0].set_xticks(range(len(dataset_names)))
    axes[1, 0].set_xticklabels(dataset_names, rotation=15, ha='right')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    for bar, score in zip(bars3, davies_scores):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{score:.4f}', ha='center', va='bottom')
    
    # 4. Išorinio vertinimo palyginimas (ARI)
    ari_scores = []
    for results in datasets_results.values():
        external = results['metrics']['external_metrics']
        if external:
            ari_scores.append(external['ari'])
        else:
            ari_scores.append(0)
    
    bars4 = axes[1, 1].bar(range(len(dataset_names)), ari_scores, 
                          color=colors[:len(dataset_names)], alpha=0.7)
    axes[1, 1].set_title('Adjusted Rand Index palyginimas')
    axes[1, 1].set_xlabel('Duomenų rinkinys')
    axes[1, 1].set_ylabel('ARI Score')
    axes[1, 1].set_xticks(range(len(dataset_names)))
    axes[1, 1].set_xticklabels(dataset_names, rotation=15, ha='right')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].set_ylim(0, 1)
    
    for bar, score in zip(bars4, ari_scores):
        height = bar.get_height()
        if score > 0:
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{score:.4f}', ha='center', va='bottom')
    
    plt.suptitle(f'K-means analizės palyginimas (k={recommended_k})', fontsize=16)
    plt.tight_layout()
    
    # Išsaugojimas
    plt.savefig('outputs/kmeans_summary_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Sukurtas bendras palyginimo grafikas")

def generate_comparison_report(datasets_results, recommended_k, optimal_results):
    """
    Generuoja išsamų palyginimo raportą
    """
    print(f"\n{'='*60}")
    print("           K-MEANS ANALIZĖS PALYGINIMO ATASKAITA")
    print(f"{'='*60}")
    
    print(f"\nOPTIMALUS K PASIRINKIMAS:")
    print(f"Empirinis metodas: k = {optimal_results['empirical']}")
    print(f"Elbow metodas: k = {optimal_results['elbow']}")
    print(f"Silhouette metodas: k = {optimal_results['silhouette']}")
    print(f"NAUDOJAMAS K: {recommended_k} (daugumos balsavimas)")
    
    # Metrikos palyginimas
    print(f"\n{'DUOMENŲ RINKINYS':<25} {'SILHOUETTE':<12} {'CALINSKI-H':<12} {'DAVIES-B':<12}")
    print("-" * 65)
    
    for dataset_name, results in datasets_results.items():
        metrics = results['metrics']
        
        print(f"{dataset_name:<25} {metrics['silhouette']:<12.4f} "
              f"{metrics['calinski_harabasz']:<12.2f} {metrics['davies_bouldin']:<12.4f}")
    
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
    
    # JSON išsaugojimas
    comparison_data = {
        'optimal_k_methods': optimal_results,
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
                'davies_bouldin': float(results['metrics']['davies_bouldin'])
            },
            'cluster_sizes': results['metrics']['cluster_counts'],
            'external_metrics': results['metrics']['external_metrics']
        }
    
    with open('outputs/kmeans_three_datasets_comparison.json', 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)
    
    # Papildomas optimal_k_results.json (suderinami su senu formatu)
    optimal_k_legacy = {
        'methods': {
            'empirical': optimal_results['empirical'],
            'elbow': optimal_results['elbow'],
            'silhouette': optimal_results['silhouette']
        },
        'recommendation': {
            'optimal_k': recommended_k,
            'consensus': 'automatic'
        }
    }
    
    with open('outputs/optimal_k_results.json', 'w', encoding='utf-8') as f:
        json.dump(optimal_k_legacy, f, indent=2, ensure_ascii=False)

def main():
    """
    Pagrindinė funkcija - analizuoja tris duomenų rinkinius
    """
    print("K-MEANS KLASTERIZACIJOS ANALIZĖ")
    print("Trys duomenų rinkiniai: Pilnas, Chi², t-SNE")
    print("="*60)
    
    os.makedirs('outputs', exist_ok=True)
    
    # 1. Parenkame optimalų k naudodami visus metodus
    print("Užkraunami duomenys optimalaus k skaičiavimui...")
    X_chi2_preview, _, _, _ = load_chi2_dataset()  # Naudojame Chi² greitumui
    optimal_results = find_optimal_k(X_chi2_preview, max_k=10)
    recommended_k = optimal_results['recommended']
    
    # 2. Paruošiame tris duomenų rinkinius
    datasets = []
    
    try:
        # 2.1 Pilna duomenų aibė
        X_full, target_full, features_full, name_full = load_full_dataset()
        datasets.append((X_full, target_full, features_full, name_full))
        
        # 2.2 Chi² atrinkti požymiai (jau turime)
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
    
    # 4. Individualūs grafikai kiekvienam duomenų rinkiniui
    visualize_individual_datasets(datasets_results, recommended_k)
    
    # 5. Bendras palyginimo grafikas
    create_summary_comparison(datasets_results, recommended_k)
    
    # 6. JSON ataskaita su optimalaus k informacija
    generate_comparison_report(datasets_results, recommended_k, optimal_results)
    
    print(f"\n=== SUKURTI PNG FAILAI ===")
    print(f"• optimal_k_methods.png - Optimalaus k metodai")
    safe_names = []
    for dataset_name in datasets_results.keys():
        safe_name = dataset_name.replace(' ', '_').replace('²', '2')
        safe_names.append(safe_name)
        print(f"• {safe_name}_clusters.png - Klasterių vizualizacija")
        print(f"• {safe_name}_metrics.png - Vertinimo metrikos")
        print(f"• {safe_name}_characteristics.png - Klasterių charakteristikos")
    
    print(f"• kmeans_summary_comparison.png - Bendras palyginimas")
    print(f"• kmeans_three_datasets_comparison.json - Duomenų ataskaita")
    print(f"• optimal_k_results.json - Optimalaus k rezultatai")
    
    print(f"\nViso sukurta: {len(safe_names) * 3 + 2} PNG failų + 2 JSON failai")
    print("Visi failai išsaugoti 'outputs/' kataloge")
    
    return datasets_results

if __name__ == "__main__":
    results = main()