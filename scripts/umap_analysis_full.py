"""
UMAP analizė su 4 konfigūracijomis
Automatiškai paleidžia visas konfigūracijas ir generuoja ataskaitą
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import umap
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from datetime import datetime

# Konfigūracijos
CONFIGS = {
    1: {'n_neighbors': 15, 'min_dist': 0.1, 'spread': 1.0, 'name': 'Detali lokali struktūra'},
    2: {'n_neighbors': 25, 'min_dist': 0.2, 'spread': 1.0, 'name': 'Subalansuota'},
    3: {'n_neighbors': 40, 'min_dist': 0.3, 'spread': 1.5, 'name': 'Globali struktūra'},
    4: {'n_neighbors': 45, 'min_dist': 0.9, 'spread': 1.0, 'name': 'Labai globali struktūra'}
}

POZYMIAI = ['Gender', 'FCVC', 'SMOKE', 'CALC', 'NCP', 'CH2O']
METRICS = ['euclidean', 'manhattan', 'cosine']

# Spalvos nutukimo tipams
CLASS_COLORS = {
    '4': "#F399FF", 4: "#F399FF",  # Rožinė (Nutukimo tipas 1)
    '5': "#0080DB", 5: "#0080DB",  # Mėlyna (Nutukimo tipas 2)
    '6': "#48A348", 6: "#48A348"   # Žalia (Nutukimo tipas 3)
}

def _minmax_from_clean(clean_df, cols):
    out = clean_df.copy()
    for c in cols:
        s = pd.to_numeric(out[c], errors='coerce')
        mn, mx = s.min(), s.max()
        if pd.isna(mn) or pd.isna(mx) or mx == mn:
            out[c] = 0.0
        else:
            out[c] = (s - mn) / (mx - mn)
    return out

def load_data(pozymiai, naudoti_raw=False):
    base_path = os.path.join(os.path.dirname(__file__), "..", "data")
    file_name = "clean_data.csv" if naudoti_raw else "normalized_minmax.csv"
    file_path = os.path.join(base_path, file_name)
    
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Failas nerastas: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"✓ Užkrautas: {file_name}")
    
    missing_cols = [c for c in pozymiai if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Trūksta požymių: {missing_cols}")
    
    df = df.dropna(subset=pozymiai)
    for c in pozymiai:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    return df

def load_outliers_and_filter(df, naudoti_raw=False):
    base_path = os.path.join(os.path.dirname(__file__), "..", "data")
    outlier_file = "non-norm_outliers.csv" if naudoti_raw else "outliers.csv"
    outlier_path = os.path.join(base_path, outlier_file)
    
    if not os.path.isfile(outlier_path):
        print(f"⚠ Išskirčių failas nerastas: {outlier_file}")
        return df, np.zeros(len(df), dtype=bool), np.zeros(len(df), dtype=bool)
    
    outliers_df = pd.read_csv(outlier_path)
    print(f"✓ Užkrautas: {outlier_file}")
    
    if 'outlier_type' not in outliers_df.columns:
        print("⚠ Trūksta 'outlier_type' stulpelio")
        return df, np.zeros(len(df), dtype=bool), np.zeros(len(df), dtype=bool)
    
    # Filtruojame pagal NObeyesdad 4,5,6
    allowed = {'4', '5', '6', 4, 5, 6}
    if 'NObeyesdad' in df.columns:
        mask = df['NObeyesdad'].astype(str).isin(allowed)
        df = df[mask].reset_index(drop=True)
        outliers_df = outliers_df[mask].reset_index(drop=True)
        print(f"✓ Filtruota pagal NObeyesdad 4,5,6: {len(df)} eilutės")
    
    # Sukuriame išskirčių maskes
    outlier_values = outliers_df['outlier_type'].values
    inner_mask = (outlier_values == 1)
    outer_mask = (outlier_values == 2)
    
    return df, inner_mask, outer_mask

def compute_metrics(embedding, labels):
    """Skaičiuoja silhouette ir purity metrikas"""
    try:
        # Silhouette score
        if len(np.unique(labels)) < 2:
            sil = np.nan
        else:
            sil = silhouette_score(embedding, labels)
        
        # Purity (KMeans)
        k = len(np.unique(labels))
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(embedding)
        
        # Skaičiuojame purity
        df_temp = pd.DataFrame({'true': labels, 'cluster': kmeans.labels_})
        grp = df_temp.groupby('cluster')['true'].value_counts()
        maj = grp.groupby(level=0).max()
        purity = maj.sum() / len(df_temp)
        
        return float(sil), float(purity)
    except Exception as e:
        print(f"⚠ Metrikų skaičiavimo klaida: {e}")
        return np.nan, np.nan

def run_umap_and_plot(df, pozymiai, config_id, config, metric, naudoti_raw, inner_mask, outer_mask, output_dir):
    """Paleidžia UMAP ir sukuria vizualizaciją"""
    X = df[pozymiai].astype(float).values
    labels_str = df['NObeyesdad'].astype(str).values if 'NObeyesdad' in df.columns else None
    labels_int = df['NObeyesdad'].astype(int).values if 'NObeyesdad' in df.columns else None
    
    # UMAP
    umap_model = umap.UMAP(
        n_neighbors=config['n_neighbors'],
        min_dist=config['min_dist'],
        spread=config['spread'],
        metric=metric,
        random_state=42,
        init='spectral',
        low_memory=True
    )
    embedding = umap_model.fit_transform(X)
    
    # Metrikos
    sil, purity = compute_metrics(embedding, labels_int) if labels_int is not None else (np.nan, np.nan)
    
    # Išskirčių statistika
    inner_count = np.sum(inner_mask)
    outer_count = np.sum(outer_mask)
    total_outliers = inner_count + outer_count
    outlier_pct = (total_outliers / len(df)) * 100 if len(df) > 0 else 0
    
    # Vizualizacija
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if labels_str is not None:
        unique_labels = sorted(set(labels_str))
        
        # Pagrindiniai taškai (ne išskirtys)
        for label in unique_labels:
            mask = (labels_str == label) & (~inner_mask) & (~outer_mask)
            color = CLASS_COLORS.get(label, CLASS_COLORS.get(str(label), '#1f77b4'))
            ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                      s=35, alpha=0.75, color=color)
        
        # Vidinės išskirtys su juodu apvadu
        if np.any(inner_mask):
            for label in unique_labels:
                mask = (labels_str == label) & inner_mask
                if np.any(mask):
                    color = CLASS_COLORS.get(label, CLASS_COLORS.get(str(label), '#1f77b4'))
                    ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                              s=60, facecolor=color, edgecolor='black', 
                              linewidth=1.2, alpha=0.95, zorder=3)
        
        # Išorinės išskirtys su raudonu apvadu
        if np.any(outer_mask):
            for label in unique_labels:
                mask = (labels_str == label) & outer_mask
                if np.any(mask):
                    color = CLASS_COLORS.get(label, CLASS_COLORS.get(str(label), '#1f77b4'))
                    ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                              s=60, facecolor=color, edgecolor='red', 
                              linewidth=1.4, alpha=0.95, zorder=4)
    
    # Legenda
    legend_elements = []
    for label in sorted(unique_labels):
        color = CLASS_COLORS.get(label, CLASS_COLORS.get(str(label), '#1f77b4'))
        legend_elements.append(
            Line2D([0], [0], marker='o', linestyle='None',
                   markerfacecolor=color, markeredgecolor='none',
                   label=f'Nutukimo tipas {label}', markersize=9)
        )
    legend_elements.append(
        Line2D([0], [0], marker='o', linestyle='None',
               markerfacecolor='white', markeredgecolor='black',
               label='Vidinė išskirtis', markersize=9, markeredgewidth=1.3)
    )
    legend_elements.append(
        Line2D([0], [0], marker='o', linestyle='None',
               markerfacecolor='white', markeredgecolor='red',
               label='Išorinė išskirtis', markersize=9, markeredgewidth=1.3)
    )
    
    data_type = "Nenormalizuoti" if naudoti_raw else "Normalizuoti (min-max)"
    title = f"UMAP Konfigūracija {config_id}: {config['name']}\n"
    title += f"{data_type} | Metrika: {metric} | n_neighbors={config['n_neighbors']}, min_dist={config['min_dist']}\n"
    title += f"Silhouette: {sil:.3f} | Purity: {purity:.3f} | Išskirtys: {total_outliers} ({outlier_pct:.1f}%)"
    
    ax.set_title(title, fontsize=11, pad=15)
    ax.set_xlabel('UMAP 1', fontsize=10)
    ax.set_ylabel('UMAP 2', fontsize=10)
    ax.legend(handles=legend_elements, title="Kategorijos", frameon=True, 
             loc='best', fontsize=9, title_fontsize=10)
    ax.grid(True, alpha=0.2)
    
    # Išsaugome
    filename = f"umap_config{config_id}_{metric}_{'raw' if naudoti_raw else 'norm'}.png"
    filepath = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Išsaugota: {filename}")
    
    return {
        'config_id': config_id,
        'config_name': config['name'],
        'n_neighbors': config['n_neighbors'],
        'min_dist': config['min_dist'],
        'spread': config['spread'],
        'metric': metric,
        'data_type': 'raw' if naudoti_raw else 'normalized',
        'n_samples': len(df),
        'silhouette': sil,
        'purity': purity,
        'inner_outliers': inner_count,
        'outer_outliers': outer_count,
        'total_outliers': total_outliers,
        'outlier_percentage': outlier_pct
    }

def generate_report(results_df, output_dir):
    """Generuoja išsamią Markdown ataskaitą"""
    report_path = os.path.join(output_dir, "UMAP_ANALIZE_ATASKAITA.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# UMAP Dimensijų Mažinimo Analizės Ataskaita\n\n")
        f.write(f"**Sugeneruota:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("---\n\n")
        f.write("## 1. Tyrimo Apžvalga\n\n")
        f.write("### Tikslas\n")
        f.write("Ištirti UMAP (Uniform Manifold Approximation and Projection) dimensijų mažinimo metodo ")
        f.write("efektyvumą nutukimo duomenų aibėje, naudojant skirtingas konfigūracijas ir parametrus.\n\n")
        
        f.write("### Duomenų Aibė\n")
        f.write("- **Požymiai:** Gender, FCVC, SMOKE, CALC, NCP, CH2O\n")
        f.write("- **Klasės:** Nutukimo tipai 4, 5, 6 (NObeyesdad)\n")
        f.write(f"- **Imties dydis:** {results_df['n_samples'].iloc[0]} įrašai\n\n")
        
        f.write("### Konfigūracijos\n\n")
        for config_id, config in CONFIGS.items():
            f.write(f"**Konfigūracija {config_id}: {config['name']}**\n")
            f.write(f"- n_neighbors: {config['n_neighbors']}\n")
            f.write(f"- min_dist: {config['min_dist']}\n")
            f.write(f"- spread: {config['spread']}\n\n")
        
        f.write("---\n\n")
        f.write("## 2. Rezultatų Suvestinė\n\n")
        
        # Metrikos lentelė
        f.write("### Metrikos pagal Konfigūraciją ir Duomenų Tipą\n\n")
        f.write("| Config | Pavadinimas | Duomenys | Metrika | Silhouette | Purity | Išskirtys (%) |\n")
        f.write("|--------|-------------|----------|---------|------------|--------|---------------|\n")
        
        for _, row in results_df.iterrows():
            data_type_lt = "Nenorm." if row['data_type'] == 'raw' else "Norm."
            f.write(f"| {row['config_id']} | {row['config_name'][:15]} | {data_type_lt} | ")
            f.write(f"{row['metric'][:4]} | {row['silhouette']:.3f} | {row['purity']:.3f} | ")
            f.write(f"{row['outlier_percentage']:.1f}% |\n")
        
        f.write("\n---\n\n")
        f.write("## 3. Geriausių Rezultatų Analizė\n\n")
        
        # Geriausi rezultatai pagal silhouette
        best_sil = results_df.loc[results_df['silhouette'].idxmax()]
        f.write(f"### Aukščiausias Silhouette Score: {best_sil['silhouette']:.3f}\n")
        f.write(f"- **Konfigūracija:** {best_sil['config_id']} ({best_sil['config_name']})\n")
        f.write(f"- **Duomenys:** {best_sil['data_type']}\n")
        f.write(f"- **Metrika:** {best_sil['metric']}\n")
        f.write(f"- **Purity:** {best_sil['purity']:.3f}\n\n")
        
        # Geriausi rezultatai pagal purity
        best_pur = results_df.loc[results_df['purity'].idxmax()]
        f.write(f"### Aukščiausias Purity Score: {best_pur['purity']:.3f}\n")
        f.write(f"- **Konfigūracija:** {best_pur['config_id']} ({best_pur['config_name']})\n")
        f.write(f"- **Duomenys:** {best_pur['data_type']}\n")
        f.write(f"- **Metrika:** {best_pur['metric']}\n")
        f.write(f"- **Silhouette:** {best_pur['silhouette']:.3f}\n\n")
        
        f.write("---\n\n")
        f.write("## 4. Konfigūracijų Palyginimas\n\n")
        
        for config_id in sorted(results_df['config_id'].unique()):
            config = CONFIGS[config_id]
            config_data = results_df[results_df['config_id'] == config_id]
            
            f.write(f"### Konfigūracija {config_id}: {config['name']}\n\n")
            f.write(f"**Parametrai:** n_neighbors={config['n_neighbors']}, ")
            f.write(f"min_dist={config['min_dist']}, spread={config['spread']}\n\n")
            
            # Vidutinės metrikos
            avg_sil = config_data['silhouette'].mean()
            avg_pur = config_data['purity'].mean()
            
            f.write(f"**Vidutinės metrikos:**\n")
            f.write(f"- Silhouette: {avg_sil:.3f}\n")
            f.write(f"- Purity: {avg_pur:.3f}\n\n")
            
            f.write("**Interpretacija:**\n")
            if config_id == 1:
                f.write("- Akcentuoja lokalias struktūras ir smulkius klasterius\n")
                f.write("- Geriausiai identifikuoja išskirtis ir lokalius santykius\n")
                f.write("- Tinka detaliai išskirčių analizei\n\n")
            elif config_id == 2:
                f.write("- Subalansuotas požiūris tarp lokalių ir globalių struktūrų\n")
                f.write("- Tinka bendram duomenų supratimui\n")
                f.write("- Optimali konfigūracija bendrai vizualizacijai\n\n")
            elif config_id == 3:
                f.write("- Pabrėžia globalias struktūras ir bendrus santykius\n")
                f.write("- Slopina lokalų triukšmą\n")
                f.write("- Tinka klasių atskyrimo vertinimui\n\n")
            else:
                f.write("- Labai aukšto lygio globali struktūra\n")
                f.write("- Gali prarasti lokalius niuansus\n")
                f.write("- Tinka tik labai bendram apžvalgai\n\n")
        
        f.write("---\n\n")
        f.write("## 5. Normalizacijos Įtaka\n\n")
        
        # Palyginimas raw vs normalized
        norm_results = results_df[results_df['data_type'] == 'normalized']
        raw_results = results_df[results_df['data_type'] == 'raw']
        
        avg_sil_norm = norm_results['silhouette'].mean()
        avg_sil_raw = raw_results['silhouette'].mean()
        avg_pur_norm = norm_results['purity'].mean()
        avg_pur_raw = raw_results['purity'].mean()
        
        f.write("### Vidutinės Metrikos\n\n")
        f.write("| Duomenų Tipas | Silhouette | Purity |\n")
        f.write("|---------------|------------|--------|\n")
        f.write(f"| Normalizuoti | {avg_sil_norm:.3f} | {avg_pur_norm:.3f} |\n")
        f.write(f"| Nenormalizuoti | {avg_sil_raw:.3f} | {avg_pur_raw:.3f} |\n\n")
        
        f.write("### Išvados\n")
        if avg_sil_norm > avg_sil_raw:
            f.write("- Normalizacija **pagerino** klasterių atskyrimo kokybę\n")
            f.write("- Min-max normalizacija padėjo suvienodinti požymių skalę\n")
        else:
            f.write("- Nenormalizuoti duomenys parodė **geresnį** atskyrumą\n")
            f.write("- Originalios skalės nešė svarbią informaciją\n")
        
        if avg_pur_norm > avg_pur_raw:
            f.write("- Normalizacija padidino klasių grynumą klasteriuose\n\n")
        else:
            f.write("- Nenormalizuoti duomenys turėjo grynesnį klasių pasiskirstymą\n\n")
        
        f.write("---\n\n")
        f.write("## 6. Išskirčių Analizė\n\n")
        
        total_inner = results_df['inner_outliers'].iloc[0]
        total_outer = results_df['outer_outliers'].iloc[0]
        total = results_df['total_outliers'].iloc[0]
        
        f.write(f"### Išskirčių Statistika\n\n")
        f.write(f"- **Vidinės išskirtys (1.5×IQR):** {total_inner} ({(total_inner/results_df['n_samples'].iloc[0]*100):.1f}%)\n")
        f.write(f"- **Išorinės išskirtys (3×IQR):** {total_outer} ({(total_outer/results_df['n_samples'].iloc[0]*100):.1f}%)\n")
        f.write(f"- **Viso išskirčių:** {total} ({(total/results_df['n_samples'].iloc[0]*100):.1f}%)\n\n")
        
        f.write("### Išskirčių Vizualizacija\n")
        f.write("- **Juodas apvadas** – vidinės išskirtys (1.5×IQR)\n")
        f.write("- **Raudonas apvadas** – išorinės išskirtys (3×IQR)\n\n")
        
        f.write("---\n\n")
        f.write("## 7. Metrikų Įtaka\n\n")
        
        for metric in METRICS:
            metric_data = results_df[results_df['metric'] == metric]
            avg_sil = metric_data['silhouette'].mean()
            avg_pur = metric_data['purity'].mean()
            
            f.write(f"### {metric.capitalize()} Metrika\n")
            f.write(f"- Vidutinis Silhouette: {avg_sil:.3f}\n")
            f.write(f"- Vidutinis Purity: {avg_pur:.3f}\n\n")
        
        f.write("---\n\n")
        f.write("## 8. Pagrindinės Išvados\n\n")
        
        best_config = results_df.loc[results_df['silhouette'].idxmax()]
        
        f.write("1. **Geriausia konfigūracija bendroms užduotims:**\n")
        f.write(f"   - Konfigūracija {best_config['config_id']} ({best_config['config_name']})\n")
        f.write(f"   - {best_config['data_type'].capitalize()} duomenys\n")
        f.write(f"   - {best_config['metric'].capitalize()} metrika\n\n")
        
        f.write("2. **Normalizacijos efektas:**\n")
        if avg_sil_norm > avg_sil_raw:
            f.write("   - Normalizacija reikšmingai pagerina rezultatus\n")
            f.write("   - Rekomenduojama naudoti min-max normalizaciją\n\n")
        else:
            f.write("   - Originali skalė išlaiko svarbią informaciją\n")
            f.write("   - Normalizacija nėra būtina šiam duomenų rinkiniui\n\n")
        
        f.write("3. **Parametrų jautrumas:**\n")
        f.write("   - Mažesnis n_neighbors (15-25) geriau išskiria lokalias struktūras\n")
        f.write("   - Didesnis n_neighbors (40-45) sukuria stabilesnę globalią struktūrą\n")
        f.write("   - min_dist <0.3 išlaiko detales, >0.5 sujungia klasterius\n\n")
        
        f.write("4. **Išskirtys:**\n")
        if total / results_df['n_samples'].iloc[0] > 0.1:
            f.write("   - Didelis išskirčių kiekis (>10%) rodo duomenų heterogeniškumą\n")
            f.write("   - Išskirtys neformuoja atskiro klasterio, o yra pasklidę\n\n")
        else:
            f.write("   - Santykinis išskirčių kiekis yra mažas (<10%)\n")
            f.write("   - Pagrindinė duomenų masė yra homogeniška\n\n")
        
        f.write("---\n\n")
        f.write("## 9. Rekomendacijos\n\n")
        f.write("### Tolimesniems Tyrimams\n")
        f.write("1. Išbandyti kitas metrikos (mahalanobis, correlation)\n")
        f.write("2. Atlikti stabilumo analizę su skirtingais random_state\n")
        f.write("3. Palyginti su kitais dimensijų mažinimo metodais (t-SNE, PCA)\n")
        f.write("4. Ištirti išskirčių pobūdį – ar tai klaidos, ar realios ypatingos būsenos\n\n")
        
        f.write("### Praktiniam Taikymui\n")
        f.write(f"1. Naudoti Konfigūraciją {best_config['config_id']} bendrai vizualizacijai\n")
        f.write("2. Konfigūracija 1 – išskirčių identifikavimui\n")
        f.write("3. Konfigūracija 3-4 – klasių separavimo vertinimui\n\n")
        
        f.write("---\n\n")
        f.write("## 10. Vizualizacijos\n\n")
        f.write("Visos vizualizacijos išsaugotos `outputs/umap/` kataloge.\n\n")
        f.write("Pavadinimų formatas: `umap_config{N}_{metrika}_{raw|norm}.png`\n\n")
        
        f.write("**Pavyzdžiai:**\n")
        f.write("- `umap_config1_euclidean_norm.png` – Konfigūracija 1, Euclidean, normalizuoti\n")
        f.write("- `umap_config2_manhattan_raw.png` – Konfigūracija 2, Manhattan, nenormalizuoti\n\n")
        
        f.write("---\n\n")
        f.write("*Ataskaita sugeneruota automatiškai naudojant `umap_analysis_full.py`*\n")
    
    print(f"\n✓ Ataskaita išsaugota: {report_path}")

def main():
    print("="*70)
    print("UMAP PILNA ANALIZĖ SU 4 KONFIGŪRACIJOMIS")
    print("="*70)
    
    # Sukuriame output katalogą
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "umap")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n✓ Output katalogas: {output_dir}\n")
    
    results = []
    
    # Paleidžiame kiekvieną kombinaciją
    for data_type in ['raw', 'normalized']:
        naudoti_raw = (data_type == 'raw')
        print(f"\n{'='*70}")
        print(f"Duomenų tipas: {'Nenormalizuoti (clean_data.csv)' if naudoti_raw else 'Normalizuoti (normalized_minmax.csv)'}")
        print(f"{'='*70}\n")
        
        # Užkrauname duomenis
        df = load_data(POZYMIAI, naudoti_raw)
        df, inner_mask, outer_mask = load_outliers_and_filter(df, naudoti_raw)
        
        for config_id, config in CONFIGS.items():
            print(f"\n--- Konfigūracija {config_id}: {config['name']} ---")
            
            for metric in METRICS:
                print(f"  → Metrika: {metric}")
                result = run_umap_and_plot(
                    df, POZYMIAI, config_id, config, metric,
                    naudoti_raw, inner_mask, outer_mask, output_dir
                )
                results.append(result)
    
    # Sukuriame rezultatų DataFrame
    results_df = pd.DataFrame(results)
    
    # Išsaugome CSV
    csv_path = os.path.join(output_dir, "umap_results_summary.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓ Rezultatų lentelė išsaugota: {csv_path}")
    
    # Generuojame ataskaitą
    print("\n" + "="*70)
    print("Generuojama ataskaita...")
    print("="*70)
    generate_report(results_df, output_dir)
    
    print("\n" + "="*70)
    print("ANALIZĖ BAIGTA SĖKMINGAI!")
    print("="*70)
    print(f"\nViso sugeneruota:")
    print(f"  - {len(results)} vizualizacijų (PNG)")
    print(f"  - 1 rezultatų lentelė (CSV)")
    print(f"  - 1 išsami ataskaita (Markdown)")
    print(f"\nVisi failai: {output_dir}")

if __name__ == '__main__':
    main()
