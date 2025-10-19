# UMAP Dimensijų Mažinimo Analizės Ataskaita

**Sugeneruota:** 2025-10-18 13:09:52

---

## 1. Tyrimo Apžvalga

### Tikslas
Ištirti UMAP (Uniform Manifold Approximation and Projection) dimensijų mažinimo metodo efektyvumą nutukimo duomenų aibėje, naudojant skirtingas konfigūracijas ir parametrus.

### Duomenų Aibė
- **Požymiai:** Gender, FCVC, SMOKE, CALC, NCP, CH2O
- **Klasės:** Nutukimo tipai 4, 5, 6 (NObeyesdad)
- **Imties dydis:** 972 įrašai

### Konfigūracijos

**Konfigūracija 1: Detali lokali struktūra**
- n_neighbors: 15
- min_dist: 0.1
- spread: 1.0

**Konfigūracija 2: Subalansuota**
- n_neighbors: 25
- min_dist: 0.2
- spread: 1.0

**Konfigūracija 3: Globali struktūra**
- n_neighbors: 40
- min_dist: 0.3
- spread: 1.5

**Konfigūracija 4: Labai globali struktūra**
- n_neighbors: 45
- min_dist: 0.9
- spread: 1.0

---

## 2. Rezultatų Suvestinė

### Metrikos pagal Konfigūraciją ir Duomenų Tipą

| Config | Pavadinimas | Duomenys | Metrika | Silhouette | Purity | Išskirtys (%) |
|--------|-------------|----------|---------|------------|--------|---------------|
| 1 | Detali lokali s | Nenorm. | eucl | 0.022 | 0.573 | 18.7% |
| 1 | Detali lokali s | Nenorm. | manh | 0.075 | 0.623 | 18.7% |
| 1 | Detali lokali s | Nenorm. | cosi | 0.086 | 0.524 | 18.7% |
| 2 | Subalansuota | Nenorm. | eucl | 0.116 | 0.611 | 18.7% |
| 2 | Subalansuota | Nenorm. | manh | 0.168 | 0.610 | 18.7% |
| 2 | Subalansuota | Nenorm. | cosi | 0.121 | 0.630 | 18.7% |
| 3 | Globali struktū | Nenorm. | eucl | 0.039 | 0.605 | 18.7% |
| 3 | Globali struktū | Nenorm. | manh | 0.142 | 0.591 | 18.7% |
| 3 | Globali struktū | Nenorm. | cosi | 0.104 | 0.587 | 18.7% |
| 4 | Labai globali s | Nenorm. | eucl | 0.179 | 0.731 | 18.7% |
| 4 | Labai globali s | Nenorm. | manh | 0.116 | 0.653 | 18.7% |
| 4 | Labai globali s | Nenorm. | cosi | 0.063 | 0.572 | 18.7% |
| 1 | Detali lokali s | Norm. | eucl | 0.145 | 0.651 | 18.6% |
| 1 | Detali lokali s | Norm. | manh | 0.075 | 0.523 | 18.6% |
| 1 | Detali lokali s | Norm. | cosi | 0.094 | 0.620 | 18.6% |
| 2 | Subalansuota | Norm. | eucl | 0.117 | 0.570 | 18.6% |
| 2 | Subalansuota | Norm. | manh | 0.156 | 0.636 | 18.6% |
| 2 | Subalansuota | Norm. | cosi | 0.112 | 0.583 | 18.6% |
| 3 | Globali struktū | Norm. | eucl | 0.113 | 0.629 | 18.6% |
| 3 | Globali struktū | Norm. | manh | 0.102 | 0.636 | 18.6% |
| 3 | Globali struktū | Norm. | cosi | 0.111 | 0.584 | 18.6% |
| 4 | Labai globali s | Norm. | eucl | 0.125 | 0.636 | 18.6% |
| 4 | Labai globali s | Norm. | manh | 0.162 | 0.622 | 18.6% |
| 4 | Labai globali s | Norm. | cosi | 0.093 | 0.636 | 18.6% |

---

## 3. Geriausių Rezultatų Analizė

### Aukščiausias Silhouette Score: 0.179
- **Konfigūracija:** 4 (Labai globali struktūra)
- **Duomenys:** raw
- **Metrika:** euclidean
- **Purity:** 0.731

### Aukščiausias Purity Score: 0.731
- **Konfigūracija:** 4 (Labai globali struktūra)
- **Duomenys:** raw
- **Metrika:** euclidean
- **Silhouette:** 0.179

---

## 4. Konfigūracijų Palyginimas

### Konfigūracija 1: Detali lokali struktūra

**Parametrai:** n_neighbors=15, min_dist=0.1, spread=1.0

**Vidutinės metrikos:**
- Silhouette: 0.083
- Purity: 0.586

**Interpretacija:**
- Akcentuoja lokalias struktūras ir smulkius klasterius
- Geriausiai identifikuoja išskirtis ir lokalius santykius
- Tinka detaliai išskirčių analizei

### Konfigūracija 2: Subalansuota

**Parametrai:** n_neighbors=25, min_dist=0.2, spread=1.0

**Vidutinės metrikos:**
- Silhouette: 0.132
- Purity: 0.607

**Interpretacija:**
- Subalansuotas požiūris tarp lokalių ir globalių struktūrų
- Tinka bendram duomenų supratimui
- Optimali konfigūracija bendrai vizualizacijai

### Konfigūracija 3: Globali struktūra

**Parametrai:** n_neighbors=40, min_dist=0.3, spread=1.5

**Vidutinės metrikos:**
- Silhouette: 0.102
- Purity: 0.605

**Interpretacija:**
- Pabrėžia globalias struktūras ir bendrus santykius
- Slopina lokalų triukšmą
- Tinka klasių atskyrimo vertinimui

### Konfigūracija 4: Labai globali struktūra

**Parametrai:** n_neighbors=45, min_dist=0.9, spread=1.0

**Vidutinės metrikos:**
- Silhouette: 0.123
- Purity: 0.642

**Interpretacija:**
- Labai aukšto lygio globali struktūra
- Gali prarasti lokalius niuansus
- Tinka tik labai bendram apžvalgai

---

## 5. Normalizacijos Įtaka

### Vidutinės Metrikos

| Duomenų Tipas | Silhouette | Purity |
|---------------|------------|--------|
| Normalizuoti | 0.117 | 0.611 |
| Nenormalizuoti | 0.102 | 0.609 |

### Išvados
- Normalizacija **pagerino** klasterių atskyrimo kokybę
- Min-max normalizacija padėjo suvienodinti požymių skalę
- Normalizacija padidino klasių grynumą klasteriuose

---

## 6. Išskirčių Analizė

### Išskirčių Statistika

- **Vidinės išskirtys (1.5×IQR):** 55 (5.7%)
- **Išorinės išskirtys (3×IQR):** 127 (13.1%)
- **Viso išskirčių:** 182 (18.7%)

### Išskirčių Vizualizacija
- **Juodas apvadas** – vidinės išskirtys (1.5×IQR)
- **Raudonas apvadas** – išorinės išskirtys (3×IQR)

---

## 7. Metrikų Įtaka

### Euclidean Metrika
- Vidutinis Silhouette: 0.107
- Vidutinis Purity: 0.626

### Manhattan Metrika
- Vidutinis Silhouette: 0.125
- Vidutinis Purity: 0.612

### Cosine Metrika
- Vidutinis Silhouette: 0.098
- Vidutinis Purity: 0.592

---

## 8. Pagrindinės Išvados

1. **Geriausia konfigūracija bendroms užduotims:**
   - Konfigūracija 4 (Labai globali struktūra)
   - Raw duomenys
   - Euclidean metrika

2. **Normalizacijos efektas:**
   - Normalizacija reikšmingai pagerina rezultatus
   - Rekomenduojama naudoti min-max normalizaciją

3. **Parametrų jautrumas:**
   - Mažesnis n_neighbors (15-25) geriau išskiria lokalias struktūras
   - Didesnis n_neighbors (40-45) sukuria stabilesnę globalią struktūrą
   - min_dist <0.3 išlaiko detales, >0.5 sujungia klasterius

4. **Išskirtys:**
   - Didelis išskirčių kiekis (>10%) rodo duomenų heterogeniškumą
   - Išskirtys neformuoja atskiro klasterio, o yra pasklidę

---

## 9. Rekomendacijos

### Tolimesniems Tyrimams
1. Išbandyti kitas metrikos (mahalanobis, correlation)
2. Atlikti stabilumo analizę su skirtingais random_state
3. Palyginti su kitais dimensijų mažinimo metodais (t-SNE, PCA)
4. Ištirti išskirčių pobūdį – ar tai klaidos, ar realios ypatingos būsenos

### Praktiniam Taikymui
1. Naudoti Konfigūraciją 4 bendrai vizualizacijai
2. Konfigūracija 1 – išskirčių identifikavimui
3. Konfigūracija 3-4 – klasių separavimo vertinimui

---

## 10. Vizualizacijos

Visos vizualizacijos išsaugotos `outputs/umap/` kataloge.

Pavadinimų formatas: `umap_config{N}_{metrika}_{raw|norm}.png`

**Pavyzdžiai:**
- `umap_config1_euclidean_norm.png` – Konfigūracija 1, Euclidean, normalizuoti
- `umap_config2_manhattan_raw.png` – Konfigūracija 2, Manhattan, nenormalizuoti

---

*Ataskaita sugeneruota automatiškai naudojant `umap_analysis_full.py`*
