# UMAP Analizės Greitas Startas

## 🚀 Greitas Paleidimas (3 Žingsniai)

### 1️⃣ Įdiekite Priklausomybes

```powershell
pip install pandas numpy matplotlib seaborn scikit-learn umap-learn
```

### 2️⃣ Paleiskite Pilną Analizę

```powershell
python scripts/umap_analysis_full.py
```

**Trukmė:** ~10-20 minučių  
**Rezultatas:** 24 vizualizacijos + metrikos + išsami ataskaita

### 3️⃣ Sukurkite Palyginimo Grafikus

```powershell
python scripts/umap_comparison_plots.py
```

**Trukmė:** ~10 sekundžių  
**Rezultatas:** 3 agregacijos grafikai

---

## 📊 Kas Bus Sugeneruota

### Vizualizacijos (24 vnt.)

Kiekviena kombinacija:

- 4 konfigūracijos × 3 metrikos × 2 duomenų tipai = 24 PNG failai

Pavyzdys:

- `umap_config1_euclidean_norm.png` ← Config 1, Euclidean, Normalizuoti
- `umap_config2_manhattan_raw.png` ← Config 2, Manhattan, Nenormalizuoti

### Metrikos

`outputs/umap/umap_results_summary.csv`:

- Silhouette score (klasterių kokybė)
- Purity score (klasių grynumas)
- Išskirčių statistika

### Ataskaitos

`outputs/umap/UMAP_ANALIZE_ATASKAITA.md`:

- Išsami kiekvienos konfigūracijos analizė
- Normalizacijos įtakos vertinimas
- Rekomendacijos tolimesniems tyrimams

### Palyginimo Grafikai

- `umap_comparison_summary.png` ← 4 grafikai (Silhouette, Purity, Metrikos, Scatter)
- `umap_heatmap_silhouette.png` ← Heatmap (Config × Metrika)
- `umap_normalization_effect.png` ← Normalizacijos efektas

---

## 🎯 Greitoji Interpretacija

### Spalvų Reikšmės

| Spalva              | Reikšmė                         |
| ------------------- | ------------------------------- |
| 🟣 Rožinė           | Nutukimo tipas 4 (lengviausias) |
| 🔵 Mėlyna           | Nutukimo tipas 5 (vidutinis)    |
| 🟢 Žalia            | Nutukimo tipas 6 (sunkiausias)  |
| ⚫ Juodas apvadas   | Vidinė išskirtis (1.5×IQR)      |
| 🔴 Raudonas apvadas | Išorinė išskirtis (3×IQR)       |

### Metrikų Interpretacija

**Silhouette Score:**

- ✅ > 0.5 = Puikiai atskirti klasteriai
- ⚠️ 0.25-0.5 = Vidutiniai
- ❌ < 0.25 = Prastas atskyrimas

**Purity Score:**

- ✅ > 0.8 = Klasteriai atitinka klases
- ⚠️ 0.6-0.8 = Vidutiniai
- ❌ < 0.6 = Klasteriai neatitinka klasių

### Konfigūracijų Pasirinkimas

| Tikslas                  | Naudoti Konfigūraciją      |
| ------------------------ | -------------------------- |
| 🔍 Išskirčių paieška     | **Config 1** (n=15, d=0.1) |
| 📊 Bendra vizualizacija  | **Config 2** (n=25, d=0.2) |
| 🌍 Klasių atskyrimas     | **Config 3** (n=40, d=0.3) |
| 🗺️ Labai bendras vaizdas | **Config 4** (n=45, d=0.9) |

---

## ⚡ Papildomi Paleidimo Būdai

### Interaktyvi Viena Konfigūracija

```powershell
python scripts/UMAP.py
```

Atsakykite į klausimus:

1. Normalizuoti? (`t`/`n`)
2. Konfigūracija? (`1`-`4`)
3. Metrika? (`euclidean`/`manhattan`/`cosine`)

### Tik Viena Konfigūracija (Programiškai)

Redaguokite `scripts/UMAP.py` ir užkomentuokite input() eilutes, nustatykite:

```python
naudoti_raw = False  # arba True
config_choice = 2
metric = 'euclidean'
```

---

## 🔧 Troubleshooting

### ❌ Klaida: "FileNotFoundError: data/clean_data.csv"

**Sprendimas:** Įsitikinkite, kad failai egzistuoja:

```powershell
ls data/clean_data.csv
ls data/normalized_minmax.csv
ls data/outliers.csv
ls data/non-norm_outliers.csv
```

### ⏳ Per lėtai vykdoma

**Sprendimas 1:** Sumažinkite konfigūracijų/metrikų skaičių  
Redaguokite `scripts/umap_analysis_full.py`:

```python
CONFIGS = {1: config_1, 2: config_2}  # Tik 2 configs
METRICS = ['euclidean']  # Tik 1 metrika
```

**Sprendimas 2:** Naudokite mažesnę imtį  
Pridėkite prieš UMAP:

```python
df = df.sample(500, random_state=42)  # 500 įrašų
```

### 💾 Per mažai RAM

**Sprendimas:** Uždarykite kitas programas arba sumažinkite n_neighbors:

```python
config_1 = {'n_neighbors': 10, ...}  # Buvo 15
```

---

## 📝 Ataskaitų Struktūra

### UMAP_ANALIZE_ATASKAITA.md Turinys

1. **Tyrimo Apžvalga** – tikslas, duomenys, konfigūracijos
2. **Rezultatų Suvestinė** – metrikos lentelė
3. **Geriausių Rezultatų Analizė** – top konfigūracijos
4. **Konfigūracijų Palyginimas** – kiekvienos detali analizė
5. **Normalizacijos Įtaka** – raw vs normalized
6. **Išskirčių Analizė** – statistika ir interpretacija
7. **Metrikų Įtaka** – euclidean vs manhattan vs cosine
8. **Pagrindinės Išvados** – santrauka
9. **Rekomendacijos** – tolimesniems tyrimams
10. **Vizualizacijos** – failų sąrašas

---

## 🎓 Kas Toliau?

### Po Analizės Peržiūrėkite:

1. ✅ `outputs/umap/UMAP_ANALIZE_ATASKAITA.md` – pilna ataskaita
2. ✅ `outputs/umap/umap_comparison_summary.png` – grafikai
3. ✅ `outputs/umap/umap_results_summary.csv` – metrikos
4. ✅ Individualias vizualizacijas pagal rūpintis kombinacijas

### Tolimesni Tyrimai:

- 🔬 Palyginti su t-SNE, PCA
- 🧪 Išbandyti kitas metrikus (mahalanobis, correlation)
- 📈 Stabilumo analizė (skirtingi random_state)
- 🎯 Išskirčių gilesnė analizė

---

## 📞 Pagalba

**Daugiau informacijos:** `UMAP_README.md`

**UMAP Dokumentacija:** https://umap-learn.readthedocs.io/

**Problemos?** Patikrinkite:

1. Python ≥ 3.8
2. Visi failai `data/` kataloge
3. Užtenka RAM (≥4GB rekomenduojama)
