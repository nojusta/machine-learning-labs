# UMAP Dimensijų Mažinimo Analizė

## Apžvalga

Šis projektas atlieka išsamią UMAP (Uniform Manifold Approximation and Projection) analizę nutukimo duomenų rinkinyje, testuojant 4 skirtingas konfigūracijas su normalizuotais ir nenormalizuotais duomenimis.

## Konfigūracijos

### Konfigūracija 1: Detali Lokali Struktūra

- **n_neighbors:** 15
- **min_dist:** 0.1
- **spread:** 1.0
- **Paskirtis:** Išskirčių identifikavimas, lokalių santykių tyrimas

### Konfigūracija 2: Subalansuota

- **n_neighbors:** 25
- **min_dist:** 0.2
- **spread:** 1.0
- **Paskirtis:** Bendras duomenų supratimas, optimalus balansas

### Konfigūracija 3: Globali Struktūra

- **n_neighbors:** 40
- **min_dist:** 0.3
- **spread:** 1.5
- **Paskirtis:** Klasių atskyrimo vertinimas, konteksto supratimas

### Konfigūracija 4: Labai Globali Struktūra

- **n_neighbors:** 45
- **min_dist:** 0.9
- **spread:** 1.0
- **Paskirtis:** Labai bendras vaizdas, globalios tendencijos

## Naudojimas

### 1. Pilna Automatizuota Analizė

Paleiskite pagrindinį skriptą, kuris sukurs visas vizualizacijas ir ataskaitą:

```powershell
python scripts/umap_analysis_full.py
```

**Kas vyksta:**

- Paleidžiamos visos 4 konfigūracijos
- Testuojamos 3 metrikos: euclidean, manhattan, cosine
- Analizuojami normalizuoti IR nenormalizuoti duomenys
- Generuojama 24 vizualizacijos (4 configs × 3 metrics × 2 data types)
- Skaičiuojamos Silhouette ir Purity metrikos
- Sukuriama išsami Markdown ataskaita

**Rezultatai:**

- `outputs/umap/umap_config{N}_{metric}_{raw|norm}.png` - vizualizacijos
- `outputs/umap/umap_results_summary.csv` - metrikos lentelė
- `outputs/umap/UMAP_ANALIZE_ATASKAITA.md` - išsami ataskaita

### 2. Palyginimo Grafikai

Po pagrindinės analizės sukurkite agreguo

tus palyginimo grafikus:

```powershell
python scripts/umap_comparison_plots.py
```

**Rezultatai:**

- `umap_comparison_summary.png` - 4 grafikai: Silhouette/Purity tendencijos
- `umap_heatmap_silhouette.png` - Heatmap konfigūracijų ir metrikų
- `umap_normalization_effect.png` - Normalizacijos įtakos grafikas

### 3. Interaktyvi Vizualizacija (Viena Konfigūracija)

Norint išbandyti vieną konfigūraciją interaktyviai:

```powershell
python scripts/UMAP.py
```

**Klausimų atsakymai:**

1. `Ar naudoti tik raw clean_data.csv (be normalizacijos)? (t/n):` → Įveskite `t` arba `n`
2. `Pasirinkite konfigūraciją (1 - detali, 2 - subalansuota, 3 - globali, 4 - labai globali):` → Įveskite 1-4
3. `Įveskite metriką (euclidean/manhattan/cosine):` → Įveskite metriką

## Rezultatų Interpretacija

### Metrikos

#### Silhouette Score (-1 iki 1)

- **> 0.5:** Puikus klasterių atskyrimas
- **0.25 - 0.5:** Vidutinis atskyrimas
- **< 0.25:** Prastas atskyrimas arba persidengiantys klasteriai

#### Purity Score (0 iki 1)

- **> 0.8:** Labai gryni klasteriai (atitinka tikrąsias klases)
- **0.6 - 0.8:** Vidutiniai rezultatai
- **< 0.6:** Klasteriai neatitinka klasių

### Vizualizacijų Spalvos

- 🟣 **Rožinė (#F399FF):** Nutukimo tipas 4 (Obesity Type I)
- 🔵 **Mėlyna (#0080DB):** Nutukimo tipas 5 (Obesity Type II)
- 🟢 **Žalia (#48A348):** Nutukimo tipas 6 (Obesity Type III)
- ⚫ **Juodas apvadas:** Vidinės išskirtys (1.5×IQR)
- 🔴 **Raudonas apvadas:** Išorinės išskirtys (3×IQR)

### Konfigūracijų Pasirinkimas

**Kada naudoti kiekvieną konfigūraciją:**

| Konfigūracija         | Kada Naudoti                                                                    |
| --------------------- | ------------------------------------------------------------------------------- |
| **1 - Detali**        | Ieškant išskirčių, analizuojant lokalius santykius, tiriant smulkias struktūras |
| **2 - Subalansuota**  | Bendrai vizualizacijai, pristatymams, balansui tarp lokalių ir globalių         |
| **3 - Globali**       | Klasių atskyrimo vertinimui, globalių tendencijų tyrimui                        |
| **4 - Labai Globali** | Labai aukšto lygio apžvalgai (gali prarasti detales)                            |

## Duomenų Struktūra

### Naudojami Požymiai

- **Gender:** Lytis (0=vyras, 1=moteris)
- **FCVC:** Daržovių vartojimo dažnis
- **SMOKE:** Rūkymas (0=ne, 1=taip)
- **CALC:** Alkoholio vartojimas
- **NCP:** Pagrindinių valgių skaičius
- **CH2O:** Vandens vartojimas

### Klasės (NObeyesdad)

Analizuojamos tik nutukimo klasės 4, 5, 6:

- **4:** Obesity Type I
- **5:** Obesity Type II
- **6:** Obesity Type III

## Technin

ės Detalės

### UMAP Parametrai

- **n_neighbors:** Kiek kaimynų naudoti lokalios struktūros įvertinimui

  - Mažesnis → daugiau lokalių detalių
  - Didesnis → stabilesnė globali struktūra

- **min_dist:** Minimalus atstumas tarp taškų projekcijoje

  - Mažesnis → tankesni, kompaktiški klasteriai
  - Didesnis → laisvesnė, sklidesnė struktūra

- **spread:** Kaip plačiai paskleisti taškai projekcijoje

  - Kontroliuoja embedding'o "plotį"

- **metric:** Atstumo matavimo metodas
  - **euclidean:** Standartinis tiesinis atstumas
  - **manhattan:** Suma absoliučių skirtumų
  - **cosine:** Kampinis panašumas (naudoja vektorių kryptis)

### Išskirčių Nustatymas

- **Vidinės išskirtys:** Q1 - 1.5×IQR arba Q3 + 1.5×IQR
- **Išorinės išskirtys:** Q1 - 3×IQR arba Q3 + 3×IQR

Kur:

- Q1 = 1-asis kvartilis
- Q3 = 3-iasis kvartilis
- IQR = Q3 - Q1 (tarpkvartilinis intervalas)

## Failų Struktūra

```
machine-learning-labs/
├── data/
│   ├── clean_data.csv              # Nenormalizuoti duomenys
│   ├── normalized_minmax.csv       # Min-max normalizuoti duomenys
│   ├── outliers.csv                # Išskirtys (normalizuotiems)
│   └── non-norm_outliers.csv       # Išskirtys (nenormalizuotiems)
├── scripts/
│   ├── UMAP.py                     # Interaktyvi vizualizacija
│   ├── umap_analysis_full.py       # Pilna automatizuota analizė
│   └── umap_comparison_plots.py    # Palyginimo grafikai
└── outputs/
    └── umap/
        ├── umap_config*.png        # Vizualizacijos
        ├── umap_results_summary.csv # Rezultatų lentelė
        ├── UMAP_ANALIZE_ATASKAITA.md # Išsami ataskaita
        └── umap_comparison_*.png   # Palyginimo grafikai
```

## Dažniausiai Užduodami Klausimai

### K: Kuri konfigūracija geriausia?

**A:** Priklauso nuo tikslo:

- Bendrai vizualizacijai → **Konfigūracija 2**
- Išskirčių tyrimui → **Konfigūracija 1**
- Klasių atskyrimui → **Konfigūracija 3**

### K: Ar reikia normalizuoti duomenis?

**A:** Palyginkite rezultatus (Silhouette/Purity) tarp `raw` ir `norm`:

- Jei `norm` geriau → naudokite normalizuotus
- Jei `raw` geriau → originali skalė yra informatyvi

### K: Kodėl daug išskirčių?

**A:** Gali būti kelios priežastys:

- Tikrai ekstremalios reikšmės (pvz., labai didelis svoris)
- Heterogeniškas duomenų rinkinys
- Maža imtis tam tikrose klasėse

### K: Kaip interpretuoti vizualizaciją?

**A:**

- Artimi taškai → panašūs objektai
- Atskiri klasteriai → skirtingos grupės
- Persidengiantys klasteriai → sunku atskirti klases
- Išskirtys pakraščiuose → atokesnės nuo pagrindinės masės

## Reikalavimai

```bash
pip install pandas numpy matplotlib seaborn scikit-learn umap-learn
```

## Troubleshooting

### Klaida: "FileNotFoundError"

- Patikrinkite, ar egzistuoja `data/clean_data.csv` ir `data/normalized_minmax.csv`
- Įsitikinkite, kad paleidžiate skriptą iš projekto root katalogo

### Klaida: "MemoryError"

- Sumažinkite n_neighbors reikšmę
- Naudokite `low_memory=True` (jau nustatyta)
- Filtruokite duomenis prieš UMAP

### Lėtas vykdymas

- Normalu, UMAP yra skaičiavimų reiklus metodas
- Konfigūracija 1-2 → ~10-30s vienam embedding'ui
- Konfigūracija 3-4 → ~30-60s vienam embedding'ui
- Pilna analizė (24 embeddings) → ~10-20 min

## Autoriai ir Licencija

Projektas sukurtas mokymosi ir tyrimo tikslais.

## Papildoma Literatūra

- [UMAP Dokumentacija](https://umap-learn.readthedocs.io/)
- [UMAP Straipsnis](https://arxiv.org/abs/1802.03426)
- [Dimensijų Mažinimo Palyginimas](https://pair-code.github.io/understanding-umap/)
