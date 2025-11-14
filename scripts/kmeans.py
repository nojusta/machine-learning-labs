import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.stats import mode
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
import matplotlib.patches as patches
import os
import warnings
warnings.filterwarnings('ignore')

# Spalvytes klasteriams musu 😍
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

def load_all_datasets():
    """Užkrauna visus duomenų rinkinius su bendromis t-SNE projekcijomis"""
    print("=== DUOMENŲ UŽKROVIMAS ===")
    
    datasets = {}
    
    # 1. Pilna normuota aibe (16 požymiu)
    try:
        df_full = pd.read_csv("./data/normalized_minmax_all.csv")
        X_full = df_full.drop(columns=["NObeyesdad"]).values
        
        # SUKURIAME t-SNE PROJEKCIJĄ PILNAI AIBEI
        print("Kuriama pilnos aibės t-SNE projekcija...")
        tsne_full = TSNE(n_components=2, perplexity=40, max_iter=600, random_state=42)
        X_full_tsne = tsne_full.fit_transform(X_full)
        
        # PASIRINKIMAS: klasterizuoti originalius duomenis ar t-SNE?
        # Variantas A: originalūs duomenys (dabartinis)
        scaler = StandardScaler()
        X_full_scaled = scaler.fit_transform(X_full)
        
        # Variantas B: t-SNE duomenys (kaip "tsne" kategorija)
        # X_full_scaled = X_full_tsne
        
        datasets['full'] = {
            'X': X_full_scaled,      # Klasterizacijai (originalus arba t-SNE)
            'X_2d': X_full_tsne,     # Vizualizacijai (visada t-SNE)
            'name': 'Pilna aibė (16 požymių)',
            'features': df_full.drop(columns=["NObeyesdad"]).columns.tolist()
        }
        print(f"Pilna aibė: {X_full_scaled.shape}, t-SNE: {X_full_tsne.shape}")
    except:
        print("Nepavyko užkrauti pilnos aibės")
    
    # 2. Atrinkti 6 pozymiai
    try:
        df_sel = pd.read_csv("./data/normalized_minmax.csv")
        X_sel = df_sel.drop(columns=["NObeyesdad"]).values
        
        # SUKURIAME t-SNE PROJEKCIJA 6 POZYMIAMS
        print("Kuriama atrinktų požymių t-SNE projekcija...")
        tsne_sel = TSNE(n_components=2, perplexity=40, max_iter=600, random_state=42)
        X_sel_tsne = tsne_sel.fit_transform(X_sel)
        
        # PASIRINKIMAS: klasterizuoti originalius duomenis ar t-SNE?
        # Variantas A: originalus duomenys
        scaler = StandardScaler()
        X_sel_scaled = scaler.fit_transform(X_sel)
        
        # Variantas B: t-SNE duomenys (isvengs overlappingo kuri gaunam keistoka)
        # X_sel_scaled = X_sel_tsne
        
        datasets['selected'] = {
            'X': X_sel_scaled,       # Klasterizacijai (originalūs arba t-SNE)
            'X_2d': X_sel_tsne,      # Vizualizacijai (visada t-SNE)
            'name': 'Atrinkti 6 požymiai',
            'features': df_sel.drop(columns=["NObeyesdad"]).columns.tolist()
        }
        print(f"Atrinkti požymiai: {X_sel_scaled.shape}, t-SNE: {X_sel_tsne.shape}")
    except:
        print("Nepavyko užkrauti atrinktų požymių")
    
    # 3. t-SNE 2D erdvė (SITAS JAU VEIKIA GERAI!)
    try:
        df_sel = pd.read_csv("./data/normalized_minmax.csv")
        X_sel_for_tsne = df_sel.drop(columns=["NObeyesdad"]).values
        
        print("Kuriama t-SNE 2D projekcija klasterizacijai...")
        tsne_for_dataset = TSNE(n_components=2, perplexity=40, max_iter=600, random_state=42)
        X_tsne = tsne_for_dataset.fit_transform(X_sel_for_tsne)
        
        datasets['tsne'] = {
            'X': X_tsne,        # Klasterizacijai (t-SNE 2D)
            'X_2d': X_tsne,     # Vizualizacijai (t-SNE 2D) - TAS PATS!
            'name': 't-SNE 2D erdvė',
            'features': ['t-SNE 1', 't-SNE 2']
        }
        print(f"t-SNE erdvė: {X_tsne.shape}")
    except Exception as e:
        print(f"Nepavyko sukurti t-SNE projekcijos: {e}")
    
    # 4. Klasterizuoti t-SNE erdvėse (cia kaip papildomas paziurejimas ar gerai, bet kiek supratau, kad neverta)
    try:
        df_sel = pd.read_csv("./data/normalized_minmax.csv")
        X_sel = df_sel.drop(columns=["NObeyesdad"]).values
        
        # t-SNE projekcija 6 požymiams
        print("Kuriama t-SNE projekcija klasterizacijai (6 požymiai)...")
        tsne_6d = TSNE(n_components=2, perplexity=40, max_iter=600, random_state=42)
        X_tsne_6d = tsne_6d.fit_transform(X_sel)
        
        datasets['selected_tsne'] = {
            'X': X_tsne_6d,         # Klasterizacijai (t-SNE 2D)
            'X_2d': X_tsne_6d,      # Vizualizacijai (t-SNE 2D) - TAS PATS vel
            'name': 'Atrinkti 6 pož. (t-SNE klasterizacija)',
            'features': ['t-SNE 1', 't-SNE 2']
        }
        print(f"6 požymiai t-SNE klasterizacijai: {X_tsne_6d.shape}")
    except:
        print("Nepavyko sukurti t-SNE klasterizacijos 6 požymiams")
    
    # 5. 16 požymių t-SNE klasterizacija
    try:
        df_full = pd.read_csv("./data/normalized_minmax_all.csv")
        X_full = df_full.drop(columns=["NObeyesdad"]).values
        
        # t-SNE projekcija 16 pozymiams (visam bliudi)
        print("Kuriama t-SNE projekcija klasterizacijai (16 požymiai)...")
        tsne_16d = TSNE(n_components=2, perplexity=40, max_iter=600, random_state=42)
        X_tsne_16d = tsne_16d.fit_transform(X_full)
        
        datasets['full_tsne'] = {
            'X': X_tsne_16d,        # Klasterizacijai (t-SNE 2D)
            'X_2d': X_tsne_16d,     # Vizualizacijai (t-SNE 2D) - TAS PATS!
            'name': 'Pilna aibė (t-SNE klasterizacija)',
            'features': ['t-SNE 1', 't-SNE 2']
        }
        print(f"16 požymių t-SNE klasterizacijai: {X_tsne_16d.shape}")
    except:
        print("Nepavyko sukurti t-SNE klasterizacijos 16 požymiams")
    
    # Tikros klases (kaip anksciau)
    try:
        df_sel = pd.read_csv("./data/normalized_minmax.csv")
        y_true = df_sel["NObeyesdad"].values
        datasets['true_labels'] = y_true
        print(f"Tikros klasės: {len(y_true)} objektai, klasės {sorted(np.unique(y_true))}")
    except:
        print("Nepavyko užkrauti tikrų klasių")
    
    return datasets

def find_optimal_k(X, max_k=8):
    """Randa optimalų k (empirinis, elbow, silhouette)"""
    print(f"\n=== OPTIMALAUS K RADIMAS ===")
    
    n_samples = X.shape[0]
    max_k = min(max_k, n_samples // 100)
    if max_k < 5:
        max_k = 5
    
    print(f"Testuojamas k diapazonas: 2-{max_k}")
    
    k_range = range(2, max_k + 1)
    sse_values = []
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        sse_values.append(kmeans.inertia_)
        sil_score = silhouette_score(X, labels)
        silhouette_scores.append(sil_score)
        
        print(f"k={k}: SSE={kmeans.inertia_:.1f}, Silhouette={sil_score:.4f}")
    
    # Trys metodai
    k_empirical = max(2, min(int(np.sqrt(n_samples / 2)), max_k))
    
    if len(sse_values) > 2:
        second_derivatives = np.diff(sse_values, 2)
        k_elbow = k_range[np.argmax(second_derivatives) + 1]
    else:
        k_elbow = 3
    
    k_silhouette = k_range[np.argmax(silhouette_scores)]
    
    # Prioriteto sistema vietoj daugumos balso
    if k_silhouette <= max_k // 2:
        recommended_k = k_silhouette
    elif k_elbow <= max_k // 2:
        recommended_k = k_elbow
    else:
        recommended_k = min(k_silhouette, k_elbow, max_k // 2)
    
    print(f"\nEmpirinis metodas: k={k_empirical}")
    print(f"Elbow metodas: k={k_elbow}")
    print(f"Silhouette metodas: k={k_silhouette}")
    print(f"REKOMENDUOJAMAS: k={recommended_k}")
    
    return recommended_k, k_range, sse_values, silhouette_scores, k_elbow, k_silhouette

def perform_kmeans_clustering(X, n_clusters):
    """K-means klasterizacija"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    cluster_labels = kmeans.fit_predict(X)
    return kmeans, cluster_labels

def map_clusters_to_classes(cluster_labels, true_labels):
    """Priskiria kiekvienam klasteriui dažniausią klasę"""
    from collections import Counter
    
    mapping = {}
    for cluster_id in np.unique(cluster_labels):
        mask = (cluster_labels == cluster_id)
        cluster_true_labels = true_labels[mask]
        
        if len(cluster_true_labels) > 0:
            label_counts = Counter(cluster_true_labels)
            most_common_label = label_counts.most_common(1)[0][0]
            mapping[cluster_id] = most_common_label
    
    mapped_preds = np.array([mapping[c] for c in cluster_labels])
    return mapped_preds, mapping

def compute_mismatch_per_class(true_labels, mapped_preds):
    """Suskaičiuoja neatitikimus kiekvienoje klasėje"""
    mismatch_counts = {}
    classes = np.unique(true_labels)
    for cls in classes:
        mask = (true_labels == cls)
        mismatches = np.sum(true_labels[mask] != mapped_preds[mask])
        mismatch_counts[int(cls)] = int(mismatches)
    return mismatch_counts

def evaluate_clustering_detailed(X, cluster_labels, y_true, dataset_name):
    """Detalus klasterizacijos vertinimas (kaip hierarchical clustering)"""
    print(f"\n=== {dataset_name.upper()} VERTINIMAS ===")
    
    # Vidine metrika
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(f"Silhouette Score: {silhouette_avg:.4f}")
    
    # Klasteriu dydziai
    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    print(f"\nKlasterių dydžiai:")
    for cluster_id, count in zip(unique_clusters, counts):
        percentage = (count / len(cluster_labels)) * 100
        print(f"  Klasteris {cluster_id}: {count} objektai ({percentage:.1f}%)")
    
    # Klasiu pasiskirstymas klasteriuose
    print(f"\nKlasės pagal klasterius:")
    for cluster_id in unique_clusters:
        mask = (cluster_labels == cluster_id)
        cluster_classes = y_true[mask]
        class_counts = pd.Series(cluster_classes).value_counts().sort_index()
        
        parts = []
        for cls, cnt in class_counts.items():
            parts.append(f"klasė {cls}: {cnt}")
        print(f"  Klasteris {cluster_id}: {', '.join(parts)}")
    
    # Tikslumas su klasteriu priskyrimu
    mapped_preds, cluster_mapping = map_clusters_to_classes(cluster_labels, y_true)
    accuracy = accuracy_score(y_true, mapped_preds)
    
    # Isorines metrikos
    ari_score = adjusted_rand_score(y_true, cluster_labels)
    nmi_score = normalized_mutual_info_score(y_true, cluster_labels)
    
    # Neatitikimu analize
    total_mismatches = np.sum(y_true != mapped_preds)
    mismatch_rate = (total_mismatches / len(y_true)) * 100
    mismatch_per_class = compute_mismatch_per_class(y_true, mapped_preds)
    
    print(f"\nIšorinis vertinimas:")
    print(f"Tikslumas: {accuracy*100:.2f}%")
    print(f"Adjusted Rand Index: {ari_score:.4f}")
    print(f"Normalized Mutual Information: {nmi_score:.4f}")
    print(f"Neatitinkančių objektų: {total_mismatches} iš {len(y_true)} ({mismatch_rate:.2f}%)")
    print(f"Klasterių → klasių atvaizdavimas: {cluster_mapping}")
    
    print(f"\nNeatitikimai pagal klasę:")
    for cls, cnt in mismatch_per_class.items():
        cls_total = np.sum(y_true == cls)
        cls_rate = (cnt / cls_total * 100) if cls_total > 0 else 0
        print(f"  Klasė {cls}: {cnt} neatitikimų iš {cls_total} ({cls_rate:.1f}%)")
    
    return {
        'silhouette': silhouette_avg,
        'accuracy': accuracy,
        'ari': ari_score,
        'nmi': nmi_score,
        'mapped_preds': mapped_preds,
        'cluster_mapping': cluster_mapping,
        'mismatch_per_class': mismatch_per_class
    }

def create_optimal_k_comparison(results_dict):
    """Optimalaus k metodų palyginimas"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    dataset_names = ['full', 'selected', 'tsne']
    titles = ['Pilna aibė (16 pož.)', 'Atrinkti 6 pož.', 't-SNE 2D erdvė']
    
    for i, (dataset_key, title) in enumerate(zip(dataset_names, titles)):
        if dataset_key not in results_dict:
            continue
            
        result = results_dict[dataset_key]
        
        # Elbow grafikas
        axes[0, i].plot(result['k_range'], result['sse_values'], 'bo-', linewidth=2, markersize=8)
        axes[0, i].axvline(x=result['k_elbow'], color='red', linestyle='--', alpha=0.8)
        axes[0, i].set_title(f'{title}\nElbow: k={result["k_elbow"]}', fontweight='bold')
        axes[0, i].set_xlabel('k')
        axes[0, i].set_ylabel('SSE')
        axes[0, i].grid(True, alpha=0.3)
        
        # Silhouette grafikas
        axes[1, i].plot(result['k_range'], result['silhouette_scores'], 'go-', linewidth=2, markersize=8)
        axes[1, i].axvline(x=result['k_silhouette'], color='red', linestyle='--', alpha=0.8)
        axes[1, i].set_title(f'Silhouette: k={result["k_silhouette"]}\nRekomenduojama: k={result["optimal_k"]}', fontweight='bold')
        axes[1, i].set_xlabel('k')
        axes[1, i].set_ylabel('Silhouette Score')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.suptitle('K-means optimalaus k metodai', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/kmean/kmeans_optimal_k_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Išsaugota: outputs/kmean/kmeans_optimal_k_comparison.png")

def create_individual_clustering_visualization(dataset_key, result, datasets, filename):
    """Sukuria klasterizacijos vizualizaciją - DABAR SU TEISINGAIS DUOMENIMIS"""
    y_true = datasets['true_labels']
    
    # DABAR VISADA NAUDOJAME PRE-COMPUTED t-SNE (BENT TURETU VEIKTI GERAI)
    X_2d = datasets[dataset_key]['X_2d']
    print(f"Naudojama iš anksto sukurta t-SNE projekcija: {datasets[dataset_key]['name']}")
    
    cluster_labels = result['cluster_labels']
    k = result['optimal_k']
    mapped_preds = result['metrics']['mapped_preds']
    accuracy = result['metrics']['accuracy']
    
    # Sukuriame 1x3 grid
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # FUNKCIJA KLASTERIU KONTURU PIESIMUI
    def draw_cluster_hulls(ax, X_2d, labels, alpha=0.2):
        """Piešia klasterių konveksinius korpusus (hulls)"""
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, cluster_id in enumerate(unique_labels):
            mask = (labels == cluster_id)
            cluster_points = X_2d[mask]
            
            if len(cluster_points) >= 3:  # Reikia bent 3 taskų hull'ui
                try:
                    hull = ConvexHull(cluster_points)
                    hull_points = cluster_points[hull.vertices]
                    
                    # Kuriame polygon'ą
                    polygon = Polygon(hull_points, alpha=alpha, 
                                    facecolor=colors[i], edgecolor=colors[i], 
                                    linewidth=2, linestyle='--')
                    ax.add_patch(polygon)
                except:
                    # Jei nepavyksta sukurti hull, piesiame apskritima
                    center = np.mean(cluster_points, axis=0)
                    radius = np.max(np.linalg.norm(cluster_points - center, axis=1)) * 1.1
                    circle = patches.Circle(center, radius, alpha=alpha, 
                                          facecolor=colors[i], edgecolor=colors[i], 
                                          linewidth=2, linestyle='--')
                    ax.add_patch(circle)
    
    # 1. Klasteriai su konturais
    draw_cluster_hulls(axes[0], X_2d, cluster_labels, alpha=0.15)
    scatter = axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_labels, 
                             cmap='tab10', alpha=0.8, s=40, edgecolors='white', linewidth=0.5)
    axes[0].set_title(f'K-means klasteriai (k={k})\nsu kontūrais', fontweight='bold')
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Tikros klases su konturais
    draw_cluster_hulls(axes[1], X_2d, y_true, alpha=0.15)
    for cls in np.unique(y_true):
        mask = (y_true == cls)
        axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1], 
                       alpha=0.8, s=40, label=f'Klasė {cls}', 
                       edgecolors='white', linewidth=0.5)
    axes[1].set_title('Tikros klasės\nsu kontūrais', fontweight='bold')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # 3. Neatitikimai su klasteriu konturais
    draw_cluster_hulls(axes[2], X_2d, cluster_labels, alpha=0.1)
    matches = (y_true == mapped_preds)
    axes[2].scatter(X_2d[matches, 0], X_2d[matches, 1], 
                   c='gray', alpha=0.6, s=25, label='Atitinka klasę',
                   edgecolors='white', linewidth=0.3)
    axes[2].scatter(X_2d[~matches, 0], X_2d[~matches, 1], 
                   c='red', alpha=0.9, s=50, label='Neatitinka klasės',
                   edgecolors='darkred', linewidth=0.8)
    
    axes[2].set_title(f'Persidengimas su klasterių erdvėmis\n(Tikslumas: {accuracy*100:.1f}%)', fontweight='bold')
    axes[2].set_xlabel('t-SNE 1')
    axes[2].set_ylabel('t-SNE 2')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f'K-means: {datasets[dataset_key]["name"]}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Išsaugota: {filename}")

def create_summary_analysis(results_dict):
    """Suvestinės analizės vizualizacija"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Tikslumu palyginimas
    dataset_names = []
    accuracies = []
    ari_scores = []
    silhouette_scores = []
    
    for key in ['full', 'selected', 'tsne']:
        if key in results_dict:
            dataset_names.append(results_dict[key]['name'])
            accuracies.append(results_dict[key]['metrics']['accuracy'] * 100)
            ari_scores.append(results_dict[key]['metrics']['ari'])
            silhouette_scores.append(results_dict[key]['metrics']['silhouette'])
    
    x_pos = np.arange(len(dataset_names))
    
    axes[0, 0].bar(x_pos, accuracies, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Klasterizacijos tikslumas', fontweight='bold')
    axes[0, 0].set_ylabel('Tikslumas (%)')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(dataset_names, rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. ARI palyginimas
    axes[0, 1].bar(x_pos, ari_scores, alpha=0.7, color='lightcoral')
    axes[0, 1].set_title('Adjusted Rand Index', fontweight='bold')
    axes[0, 1].set_ylabel('ARI')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(dataset_names, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Silhouette palyginimas
    axes[1, 0].bar(x_pos, silhouette_scores, alpha=0.7, color='lightgreen')
    axes[1, 0].set_title('Silhouette Score', fontweight='bold')
    axes[1, 0].set_ylabel('Silhouette')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(dataset_names, rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Optimaliu k rekomendaciju palyginimas
    optimal_ks = []
    for key in ['full', 'selected', 'tsne']:
        if key in results_dict:
            optimal_ks.append(results_dict[key]['optimal_k'])
    
    axes[1, 1].bar(x_pos, optimal_ks, alpha=0.7, color='gold')
    axes[1, 1].set_title('Rekomenduojami k', fontweight='bold')
    axes[1, 1].set_ylabel('Optimalus k')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(dataset_names, rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('K-means analizės suvestinė', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/kmean/kmeans_summary_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Išsaugota: outputs/kmean/kmeans_summary_analysis.png")

def create_alternative_k_visualizations(datasets, results_dict):
    """Sukuria papildomas vizualizacijas su elbow ir silhouette k visoms duomenų aibėms"""
    print(f"\n=== ALTERNATYVIŲ K VIZUALIZACIJOS ===")
    
    # Duomenu rinkiniai ir ju alternatyvus k
    alternative_configs = []
    
    for dataset_key in ['full', 'selected', 'tsne']:
        if dataset_key not in datasets or dataset_key not in results_dict:
            continue
            
        result = results_dict[dataset_key]
        k_elbow = result['k_elbow']
        k_silhouette = result['k_silhouette']
        k_optimal = result['optimal_k']
        
        # Pridesime tik jei skiriasi nuo rekomenduojamo
        if k_elbow != k_optimal or k_silhouette != k_optimal:
            alternative_configs.append({
                'dataset_key': dataset_key,
                'name': result['name'],
                'k_elbow': k_elbow,
                'k_silhouette': k_silhouette,
                'k_optimal': k_optimal
            })
    
    if not alternative_configs:
        print("Nėra alternatyvių k konfigūracijų vizualizacijai")
        return
    
    y_true = datasets['true_labels']
    
    # FUNKCIJA KLASTERIU KONTURU PIESIMUI (tas pats kaip virsuje)
    def draw_cluster_hulls(ax, X_2d, labels, alpha=0.2):
        """Piešia klasterių konveksinius korpusus (hulls)"""
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, cluster_id in enumerate(unique_labels):
            mask = (labels == cluster_id)
            cluster_points = X_2d[mask]
            
            if len(cluster_points) >= 3:  # Reikia bent 3 tašku hull'ui
                try:
                    hull = ConvexHull(cluster_points)
                    hull_points = cluster_points[hull.vertices]
                    
                    # Kuriame polygon'ą
                    polygon = Polygon(hull_points, alpha=alpha, 
                                    facecolor=colors[i], edgecolor=colors[i], 
                                    linewidth=2, linestyle='--')
                    ax.add_patch(polygon)
                except:
                    # Jei nepavyksta sukurti hull, piesiame apskritima
                    center = np.mean(cluster_points, axis=0)
                    radius = np.max(np.linalg.norm(cluster_points - center, axis=1)) * 1.1
                    circle = patches.Circle(center, radius, alpha=alpha, 
                                          facecolor=colors[i], edgecolor=colors[i], 
                                          linewidth=2, linestyle='--')
                    ax.add_patch(circle)
    
    # Kuriame vizualizacija kiekvienam duomenu rinkiniui
    for config in alternative_configs:
        dataset_key = config['dataset_key']
        name = config['name']
        k_elbow = config['k_elbow']
        k_silhouette = config['k_silhouette']
        k_optimal = config['k_optimal']
        
        print(f"\nKuriamos {name} alternatyvios vizualizacijos:")
        print(f"  Elbow k={k_elbow}, Silhouette k={k_silhouette}, Rekomenduojama k={k_optimal}")
        
        X_clustering = datasets[dataset_key]['X']  # Klasterizacijai
        X_2d = datasets[dataset_key]['X_2d']       # Vizualizacijai
        print(f"Naudojama iš anksto sukurta t-SNE projekcija: {name}")
        
        # Atliekame klasterizacijas su skirtingais k
        methods_to_test = []
        
        if k_elbow != k_optimal:
            kmeans_elbow = KMeans(n_clusters=k_elbow, random_state=42, n_init=10, max_iter=300)
            labels_elbow = kmeans_elbow.fit_predict(X_clustering)
            mapped_preds_elbow, mapping_elbow = map_clusters_to_classes(labels_elbow, y_true)
            accuracy_elbow = accuracy_score(y_true, mapped_preds_elbow)
            
            methods_to_test.append({
                'method': 'Elbow',
                'k': k_elbow,
                'labels': labels_elbow,
                'mapped_preds': mapped_preds_elbow,
                'accuracy': accuracy_elbow,
                'mapping': mapping_elbow
            })
        
        if k_silhouette != k_optimal:
            kmeans_sil = KMeans(n_clusters=k_silhouette, random_state=42, n_init=10, max_iter=300)
            labels_sil = kmeans_sil.fit_predict(X_clustering)
            mapped_preds_sil, mapping_sil = map_clusters_to_classes(labels_sil, y_true)
            accuracy_sil = accuracy_score(y_true, mapped_preds_sil)
            
            methods_to_test.append({
                'method': 'Silhouette',
                'k': k_silhouette,
                'labels': labels_sil,
                'mapped_preds': mapped_preds_sil,
                'accuracy': accuracy_sil,
                'mapping': mapping_sil
            })
        
        # Rekomenduojama k (jau turime)
        optimal_labels = results_dict[dataset_key]['cluster_labels']
        optimal_mapped_preds = results_dict[dataset_key]['metrics']['mapped_preds']
        optimal_accuracy = results_dict[dataset_key]['metrics']['accuracy']
        optimal_mapping = results_dict[dataset_key]['metrics']['cluster_mapping']
        
        methods_to_test.append({
            'method': 'Rekomenduojama',
            'k': k_optimal,
            'labels': optimal_labels,
            'mapped_preds': optimal_mapped_preds,
            'accuracy': optimal_accuracy,
            'mapping': optimal_mapping
        })
        
        # Sukuriame grid pagal metodų skaiciu
        n_methods = len(methods_to_test)
        fig, axes = plt.subplots(n_methods, 3, figsize=(18, 6 * n_methods))
        
        # Jei tik vienas metodas, axes bus 1D
        if n_methods == 1:
            axes = axes.reshape(1, -1)
        
        for i, method_data in enumerate(methods_to_test):
            method_name = method_data['method']
            k = method_data['k']
            labels = method_data['labels']
            mapped_preds = method_data['mapped_preds']
            accuracy = method_data['accuracy']
            mapping = method_data['mapping']
            
            # 1. Klasteriai su konturais
            draw_cluster_hulls(axes[i, 0], X_2d, labels, alpha=0.15)
            axes[i, 0].scatter(X_2d[:, 0], X_2d[:, 1], c=labels, 
                              cmap='tab10', alpha=0.8, s=35, edgecolors='white', linewidth=0.5)
            axes[i, 0].set_title(f'{method_name} metodas: k={k}\nKlasteriai su kontūrais', fontweight='bold')
            axes[i, 0].set_xlabel('t-SNE 1')
            axes[i, 0].set_ylabel('t-SNE 2')
            axes[i, 0].grid(True, alpha=0.3)
            
            # 2. Tikros klases (tas pats visoms eilutems) su konturais
            draw_cluster_hulls(axes[i, 1], X_2d, y_true, alpha=0.1)
            for cls in np.unique(y_true):
                mask = (y_true == cls)
                axes[i, 1].scatter(X_2d[mask, 0], X_2d[mask, 1], 
                                  alpha=0.8, s=35, label=f'Klasė {cls}',
                                  edgecolors='white', linewidth=0.5)
            axes[i, 1].set_title('Tikros klasės', fontweight='bold')
            axes[i, 1].set_xlabel('t-SNE 1')
            axes[i, 1].set_ylabel('t-SNE 2')
            axes[i, 1].legend(fontsize=9)
            axes[i, 1].grid(True, alpha=0.3)
            
            # 3. Neatitikimai su klasteriu konturais
            draw_cluster_hulls(axes[i, 2], X_2d, labels, alpha=0.1)
            matches = (y_true == mapped_preds)
            axes[i, 2].scatter(X_2d[matches, 0], X_2d[matches, 1], 
                              c='gray', alpha=0.6, s=25, label='Atitinka klasę',
                              edgecolors='white', linewidth=0.3)
            axes[i, 2].scatter(X_2d[~matches, 0], X_2d[~matches, 1], 
                              c='red', alpha=0.9, s=45, label='Neatitinka klasės',
                              edgecolors='darkred', linewidth=0.8)
            axes[i, 2].set_title(f'Persidengimas\n(Tikslumas: {accuracy*100:.1f}%)', fontweight='bold')
            axes[i, 2].set_xlabel('t-SNE 1')
            axes[i, 2].set_ylabel('t-SNE 2')
            axes[i, 2].legend(fontsize=9)
            axes[i, 2].grid(True, alpha=0.3)
            
            # Spausdiname detalizuota informacija
            print(f"  {method_name} k={k}: Tikslumas {accuracy*100:.2f}%, Atvaizdavimas {mapping}")
        
        plt.suptitle(f'{name}: Alternatyvūs K metodai su klasterių kontūrais', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Issaugome su aprasomais pavadinimais
        filename = f'outputs/kmean/kmeans_{dataset_key}_alternative_k.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Išsaugota: {filename}")

def main():
    """Pagrindinė funkcija"""
    print("K-MEANS KLASTERIZACIJOS IŠSAMI ANALIZĖ")
    print("="*60)
    
    os.makedirs('outputs/kmean', exist_ok=True)
    
    # 1. Uzkrauname visus duomenu rinkinius
    datasets = load_all_datasets()
    
    if 'true_labels' not in datasets:
        print("Klaida: nepavyko užkrauti tikrų klasių!")
        return None
    
    y_true = datasets['true_labels']
    results_dict = {}
    
    # 2. Analizuojame kiekviena duomenu rinkini
    for dataset_key in ['full', 'selected', 'tsne', 'full_tsne', 'selected_tsne']:
        if dataset_key not in datasets:
            continue
        
        dataset = datasets[dataset_key]
        X = dataset['X']
        name = dataset['name']
        
        print(f"\n{'='*20} {name.upper()} {'='*20}")
        
        # Optimalaus k radimas
        optimal_k, k_range, sse_values, sil_scores, k_elbow, k_sil = find_optimal_k(X)
        
        # K-means klasterizacija
        print(f"\n=== K-MEANS KLASTERIZACIJA (k={optimal_k}) ===")
        kmeans, cluster_labels = perform_kmeans_clustering(X, optimal_k)
        print(f"Iteracijų: {kmeans.n_iter_}, SSE: {kmeans.inertia_:.2f}")
        
        # Detalus vertinimas
        metrics = evaluate_clustering_detailed(X, cluster_labels, y_true, name)
        
        # Issaugome rezultatus
        results_dict[dataset_key] = {
            'name': name,
            'optimal_k': optimal_k,
            'k_range': list(k_range),
            'sse_values': sse_values,
            'silhouette_scores': sil_scores,
            'k_elbow': k_elbow,
            'k_silhouette': k_sil,
            'cluster_labels': cluster_labels,
            'metrics': metrics
        }
    
    # 3. Vizualizacijos
    print(f"\n=== VIZUALIZACIJŲ KŪRIMAS ===")
    
    # 3.1 Optimalaus k palyginimas (vienas failas)
    create_optimal_k_comparison(results_dict)
    
    # 3.2 Individualus klasterizacijos rezultatai (atskiri failai)
    for dataset_key, result in results_dict.items():
        filename = f"outputs/kmean/kmeans_{dataset_key}_clustering.png"
        create_individual_clustering_visualization(dataset_key, result, datasets, filename)
    
    # 3.3 Alternatyviu k vizualizacijos VISOMS duomenu aibems
    create_alternative_k_visualizations(datasets, results_dict)
    
    # 3.4 Suvestine analize (vienas failas)
    create_summary_analysis(results_dict)
    
    # 4. Suvestine
    print(f"\n{'='*20} ANALIZĖS SUVESTINĖ {'='*20}")
    for dataset_key, result in results_dict.items():
        name = result['name']
        k = result['optimal_k']
        k_elbow = result['k_elbow']
        k_sil = result['k_silhouette']
        acc = result['metrics']['accuracy'] * 100
        sil = result['metrics']['silhouette']
        
        print(f"{name}:")
        print(f"  Elbow k: {k_elbow}, Silhouette k: {k_sil}, Rekomenduojama k: {k}")
        print(f"  Tikslumas: {acc:.2f}%")
        print(f"  Silhouette Score: {sil:.4f}")
        print()
    
    print("Vizualizacijos išsaugotos 'outputs/kmean/' kataloge:")
    print("  - kmeans_optimal_k_comparison.png (visi duomenų rinkiniai)")
    print("  - kmeans_full_clustering.png (16 požymių)")
    print("  - kmeans_selected_clustering.png (6 požymiai)")
    print("  - kmeans_tsne_clustering.png (t-SNE erdvė)")
    print("  - kmeans_full_alternative_k.png (16 požymių alternatyvūs k)")
    print("  - kmeans_selected_alternative_k.png (6 požymiai alternatyvūs k)")
    print("  - kmeans_tsne_alternative_k.png (t-SNE alternatyvūs k)")
    print("  - kmeans_summary_analysis.png (suvestinė)")
    
    return results_dict

if __name__ == "__main__":
    results = main()