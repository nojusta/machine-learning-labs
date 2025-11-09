import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import os
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def find_file(filename):
    """Ieško failo keliuose"""
    candidates = [
        f"../data/{filename}",
        f"./data/{filename}", 
        f"data/{filename}",
    ]
    
    for path in candidates:
        if os.path.isfile(path):
            print(f"Rastas failas: {path}")
            return path
    
    raise FileNotFoundError(f"Nerastas failas: {filename}")

def load_data(filepath, keep_classes={4, 5, 6}, chi2_features=['Gender', 'FCVC', 'SMOKE', 'CALC', 'NCP', 'CH2O']):
    """Pakrauna duomenis"""
    df = pd.read_csv(filepath)
    df = df[df['NObeyesdad'].isin(keep_classes)].copy()
    
    target = df['NObeyesdad'].values
    X = df[chi2_features].dropna().values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, target

def find_optimal_k(X, max_k=10):
    """Randa optimalų k"""
    n = X.shape[0]
    k_emp = int(np.sqrt(n / 2))
    
    sse = []
    sil_scores = []
    K_range = range(2, min(max_k, n // 5) + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
        labels = kmeans.fit_predict(X)
        sse.append(kmeans.inertia_)
        sil_scores.append(silhouette_score(X, labels))
    
    k_elbow = K_range[np.argmin(np.diff(np.diff(sse))) + 1] if len(sse) > 2 else 3
    k_sil = K_range[np.argmax(sil_scores)]
    
    recommended_k = Counter([k_emp, k_elbow, k_sil]).most_common(1)[0][0]
    
    # Grafikas
    os.makedirs('outputs', exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.plot(K_range, sse, 'bo-')
    ax1.axvline(k_elbow, color='r', linestyle='--', label=f'k={k_elbow}')
    ax1.set_title('Elbow metodas')
    ax1.legend()
    
    ax2.plot(K_range, sil_scores, 'go-')
    ax2.axvline(k_sil, color='r', linestyle='--', label=f'k={k_sil}')
    ax2.set_title('Silhouette metodas')
    ax2.legend()
    
    plt.suptitle(f'Rekomenduojama k={recommended_k}')
    plt.tight_layout()
    plt.savefig('outputs/optimal_k.png', dpi=200)
    plt.close()
    
    return recommended_k

def run_kmeans(X, k):
    """Paleidžia K-means"""
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    
    metrics = {
        'silhouette': silhouette_score(X, labels),
        'calinski': calinski_harabasz_score(X, labels),
        'davies_bouldin': davies_bouldin_score(X, labels)
    }
    
    return kmeans, labels, metrics

def analyze_mismatches(labels, target):
    """Analizuoja neatitikimus"""
    print("\nNeatitikimai pagal klasę:")
    for class_id in np.unique(target):
        class_mask = target == class_id
        class_clusters = labels[class_mask]
        dominant_cluster = Counter(class_clusters).most_common(1)[0][0]
        dominant_count = Counter(class_clusters).most_common(1)[0][1]
        mismatches = len(class_clusters) - dominant_count
        print(f"Klasė {class_id}: {mismatches} neatitinkančių objektų")

def plot_clusters(X, labels, target, title, filename):
    """Sukuria klasterių vizualizaciją"""
    # t-SNE jei > 2D
    if X.shape[1] > 2:
        tsne = TSNE(n_components=2, perplexity=40, max_iter=600, 
                   learning_rate=200, init='pca', random_state=42)
        X_plot = tsne.fit_transform(X)
    else:
        X_plot = X
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Klasteriai
    scatter1 = axes[0].scatter(X_plot[:, 0], X_plot[:, 1], c=labels, cmap='viridis', s=30)
    axes[0].set_title('K-means klasteriai')
    
    # Tikros klasės
    if target is not None:
        scatter2 = axes[1].scatter(X_plot[:, 0], X_plot[:, 1], c=target, cmap='Set1', s=30)
        axes[1].set_title('Tikros klasės')
    
    # Klasterių dydžiai
    unique, counts = np.unique(labels, return_counts=True)
    axes[2].bar(unique, counts)
    axes[2].set_title('Klasterių dydžiai')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f'outputs/{filename}_clusters.png', dpi=200)
    plt.close()

def plot_mismatches(X, labels, target, title, filename):
    """Sukuria neatitikimų vizualizaciją"""
    if target is None:
        return
        
    # t-SNE jei > 2D
    if X.shape[1] > 2:
        tsne = TSNE(n_components=2, perplexity=40, max_iter=600, 
                   learning_rate=200, init='pca', random_state=42)
        X_plot = tsne.fit_transform(X)
    else:
        X_plot = X
    
    # Rasti neatitinkančius taškus
    mismatch_mask = np.zeros(len(target), dtype=bool)
    for class_id in np.unique(target):
        class_mask = target == class_id
        class_clusters = labels[class_mask]
        if len(class_clusters) > 0:
            dominant = Counter(class_clusters).most_common(1)[0][0]
            class_mismatch = class_mask & (labels != dominant)
            mismatch_mask |= class_mismatch
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Klasteriai
    axes[0, 0].scatter(X_plot[:, 0], X_plot[:, 1], c=labels, cmap='viridis', s=30)
    axes[0, 0].set_title('K-means klasteriai')
    
    # Tikros klasės  
    axes[0, 1].scatter(X_plot[:, 0], X_plot[:, 1], c=target, cmap='Set1', s=30)
    axes[0, 1].set_title('Tikros klasės')
    
    # Neatitikimai
    axes[1, 0].scatter(X_plot[:, 0], X_plot[:, 1], c='lightgray', s=20, alpha=0.5)
    if np.any(mismatch_mask):
        axes[1, 0].scatter(X_plot[mismatch_mask, 0], X_plot[mismatch_mask, 1], 
                          c='red', s=40, label='Neatitinka')
    axes[1, 0].set_title('Neatitikimai')
    axes[1, 0].legend()
    
    # Statistikos
    axes[1, 1].axis('off')
    total_mismatches = np.sum(mismatch_mask)
    mismatch_pct = (total_mismatches / len(target)) * 100
    
    stats_text = f"Neatitikimai: {total_mismatches}/{len(target)}\n"
    stats_text += f"Procentas: {mismatch_pct:.1f}%\n"
    stats_text += f"Tikslumas: {100-mismatch_pct:.1f}%"
    
    axes[1, 1].text(0.1, 0.7, stats_text, fontsize=12, 
                   bbox=dict(boxstyle="round", facecolor='lightyellow'))
    
    plt.suptitle(f'{title} - Neatitikimų analizė')
    plt.tight_layout()
    plt.savefig(f'outputs/{filename}_mismatches.png', dpi=200)
    plt.close()

def main():
    """Pagrindinė funkcija"""
    print("K-MEANS ANALIZĖ (SUPAPRASTINTA)")
    print("="*40)
    
    os.makedirs('outputs', exist_ok=True)
    
    # Duomenų rinkiniai su failo paieška
    try:
        datasets = [
            (find_file("normalized_minmax_all.csv"), "Pilnas"),
            (find_file("outliers.csv"), "Chi2"), 
            (find_file("outliers.csv"), "tSNE")
        ]
    except FileNotFoundError as e:
        print(f"KLAIDA: {e}")
        print("Patikrinkite ar egzistuoja failai data/ kataloge")
        return None
    
    results = {}
    
    # Rasti optimalų k
    X_temp, _ = load_data(find_file("outliers.csv"))
    optimal_k = find_optimal_k(X_temp)
    print(f"Naudojamas k = {optimal_k}")
    
    for filepath, name in datasets:
        print(f"\n--- {name} ---")
        
        # Užkrauti duomenis
        X, target = load_data(filepath)
        
        # t-SNE transformacija jei reikia
        if name == "tSNE":
            tsne = TSNE(n_components=2, perplexity=40, max_iter=600, 
                       learning_rate=200, init='pca', random_state=42)
            X = tsne.fit_transform(X)
        
        # K-means
        kmeans, labels, metrics = run_kmeans(X, optimal_k)
        
        print(f"Silhouette: {metrics['silhouette']:.4f}")
        print(f"Calinski-Harabasz: {metrics['calinski']:.2f}")
        print(f"Davies-Bouldin: {metrics['davies_bouldin']:.4f}")
        
        # Neatitikimai
        analyze_mismatches(labels, target)
        
        # Grafikai
        plot_clusters(X, labels, target, name, name.lower())
        plot_mismatches(X, labels, target, name, name.lower())
        
        results[name] = metrics
    
    # Palyginimo grafikas
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    names = list(results.keys())
    
    # Silhouette
    sil_scores = [results[name]['silhouette'] for name in names]
    axes[0].bar(names, sil_scores, color=['blue', 'green', 'red'], alpha=0.7)
    axes[0].set_title('Silhouette Score')
    axes[0].set_ylabel('Score')
    
    # Calinski-Harabasz
    cal_scores = [results[name]['calinski'] for name in names]
    axes[1].bar(names, cal_scores, color=['blue', 'green', 'red'], alpha=0.7)
    axes[1].set_title('Calinski-Harabasz')
    axes[1].set_ylabel('Score')
    
    # Davies-Bouldin
    dav_scores = [results[name]['davies_bouldin'] for name in names]
    axes[2].bar(names, dav_scores, color=['blue', 'green', 'red'], alpha=0.7)
    axes[2].set_title('Davies-Bouldin')
    axes[2].set_ylabel('Score')
    
    plt.suptitle(f'K-means palyginimas (k={optimal_k})')
    plt.tight_layout()
    plt.savefig('outputs/comparison.png', dpi=200)
    plt.close()
    
    print(f"\n=== SUKURTI FAILAI ===")
    print("outputs/optimal_k.png")
    print("outputs/pilnas_clusters.png")
    print("outputs/pilnas_mismatches.png") 
    print("outputs/chi2_clusters.png")
    print("outputs/chi2_mismatches.png")
    print("outputs/tsne_clusters.png")
    print("outputs/tsne_mismatches.png")
    print("outputs/comparison.png")
    
    return results

if __name__ == "__main__":
    results = main()