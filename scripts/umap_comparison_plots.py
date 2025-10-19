"""
UMAP konfigūracijų palyginimo grafikai
Sukuria agreguo

tas vizualizacijas rezultatų palyginimui
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_comparison_plots(results_csv, output_dir):
    """Sukuria palyginimo grafikus pagal rezultatų CSV"""
    
    df = pd.read_csv(results_csv)
    
    # 1. Silhouette ir Purity palyginimas pagal konfigūraciją
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('UMAP Konfigūracijų Palyginimas', fontsize=16, fontweight='bold')
    
    # 1a. Silhouette pagal config ir data type
    ax = axes[0, 0]
    for data_type in df['data_type'].unique():
        subset = df[df['data_type'] == data_type]
        grouped = subset.groupby('config_id')['silhouette'].mean()
        label = 'Normalizuoti' if data_type == 'normalized' else 'Nenormalizuoti'
        ax.plot(grouped.index, grouped.values, marker='o', linewidth=2, 
               markersize=8, label=label)
    ax.set_xlabel('Konfigūracija', fontweight='bold')
    ax.set_ylabel('Vidutinis Silhouette Score', fontweight='bold')
    ax.set_title('Silhouette Score pagal Konfigūraciją')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks([1, 2, 3, 4])
    
    # 1b. Purity pagal config ir data type
    ax = axes[0, 1]
    for data_type in df['data_type'].unique():
        subset = df[df['data_type'] == data_type]
        grouped = subset.groupby('config_id')['purity'].mean()
        label = 'Normalizuoti' if data_type == 'normalized' else 'Nenormalizuoti'
        ax.plot(grouped.index, grouped.values, marker='s', linewidth=2,
               markersize=8, label=label)
    ax.set_xlabel('Konfigūracija', fontweight='bold')
    ax.set_ylabel('Vidutinis Purity Score', fontweight='bold')
    ax.set_title('Purity Score pagal Konfigūraciją')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks([1, 2, 3, 4])
    
    # 1c. Metrikos palyginimas
    ax = axes[1, 0]
    metrics_grouped = df.groupby('metric')[['silhouette', 'purity']].mean()
    x = np.arange(len(metrics_grouped.index))
    width = 0.35
    ax.bar(x - width/2, metrics_grouped['silhouette'], width, label='Silhouette', alpha=0.8)
    ax.bar(x + width/2, metrics_grouped['purity'], width, label='Purity', alpha=0.8)
    ax.set_xlabel('Atstumų Metrika', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Metrikų Įtaka')
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics_grouped.index])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 1d. Scatter: Silhouette vs Purity
    ax = axes[1, 1]
    for config_id in df['config_id'].unique():
        subset = df[df['config_id'] == config_id]
        ax.scatter(subset['silhouette'], subset['purity'], 
                  s=100, alpha=0.7, label=f'Config {config_id}')
    ax.set_xlabel('Silhouette Score', fontweight='bold')
    ax.set_ylabel('Purity Score', fontweight='bold')
    ax.set_title('Silhouette vs Purity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, 'umap_comparison_summary.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Palyginimo grafikas: {comparison_path}")
    
    # 2. Heatmap: Config vs Metric (Silhouette)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, data_type in enumerate(df['data_type'].unique()):
        subset = df[df['data_type'] == data_type]
        pivot = subset.pivot_table(values='silhouette', 
                                   index='config_id', 
                                   columns='metric')
        
        ax = axes[idx]
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Silhouette Score'},
                   vmin=df['silhouette'].min(), vmax=df['silhouette'].max(),
                   ax=ax)
        title = 'Normalizuoti' if data_type == 'normalized' else 'Nenormalizuoti'
        ax.set_title(f'Silhouette Heatmap ({title})', fontweight='bold')
        ax.set_xlabel('Metrika', fontweight='bold')
        ax.set_ylabel('Konfigūracija', fontweight='bold')
    
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, 'umap_heatmap_silhouette.png')
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Heatmap grafikas: {heatmap_path}")
    
    # 3. Normalizacijos efektas
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    norm_avg = df[df['data_type'] == 'normalized'].groupby('config_id')[['silhouette', 'purity']].mean()
    raw_avg = df[df['data_type'] == 'raw'].groupby('config_id')[['silhouette', 'purity']].mean()
    
    x = np.arange(len(norm_avg.index))
    width = 0.35
    
    ax.bar(x - width/2, norm_avg['silhouette'], width, 
          label='Norm. Silhouette', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, raw_avg['silhouette'], width,
          label='Raw Silhouette', alpha=0.8, color='coral')
    
    ax.set_xlabel('Konfigūracija', fontweight='bold')
    ax.set_ylabel('Silhouette Score', fontweight='bold')
    ax.set_title('Normalizacijos Efektas (Silhouette)', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Config {i}' for i in norm_avg.index])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    norm_effect_path = os.path.join(output_dir, 'umap_normalization_effect.png')
    plt.savefig(norm_effect_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Normalizacijos efekto grafikas: {norm_effect_path}")
    
    # 4. Top konfigūracijos lentelė
    print("\n" + "="*70)
    print("TOP 5 KONFIGŪRACIJOS (pagal Silhouette)")
    print("="*70)
    top5 = df.nlargest(5, 'silhouette')[['config_id', 'config_name', 'data_type', 
                                          'metric', 'silhouette', 'purity']]
    print(top5.to_string(index=False))
    print()

if __name__ == '__main__':
    base_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "umap")
    results_csv = os.path.join(base_path, "umap_results_summary.csv")
    
    if not os.path.isfile(results_csv):
        print(f"KLAIDA: Rezultatų failas nerastas: {results_csv}")
        print("Pirma paleiskite: python scripts/umap_analysis_full.py")
    else:
        print("Kuriami palyginimo grafikai...\n")
        create_comparison_plots(results_csv, base_path)
        print("\n✓ Visi palyginimo grafikai sukurti sėkmingai!")
