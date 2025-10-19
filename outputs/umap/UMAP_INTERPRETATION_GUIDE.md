# UMAP Rezultatų Interpretavimo Vadovas

## Turinys

1. [Kaip Skaityti Vizualizacijas](#kaip-skaityti-vizualizacijas)
2. [Metrikų Interpretacija](#metrikų-interpretacija)
3. [Konfigūracijų Palyginimas](#konfigūracijų-palyginimas)
4. [Normalizacijos Efektas](#normalizacijos-efektas)
5. [Išskirčių Analizė](#išskirčių-analizė)
6. [Praktiniai Pavyzdžiai](#praktiniai-pavyzdžiai)
7. [Dažniausios Klaidos](#dažniausios-klaidos)

---

## Kaip Skaityti Vizualizacijas

### Pagrindiniai Elementai

#### 1. Taškai (Scatter Points)

Kiekvienas taškas = vienas tyrimo objektas (asmuo)

**Spalvos:**

- 🟣 **Rožinė** → Nutukimo tipas 4 (Obesity Type I - lengviausias)
- 🔵 **Mėlyna** → Nutukimo tipas 5 (Obesity Type II - vidutinis)
- 🟢 **Žalia** → Nutukimo tipas 6 (Obesity Type III - sunkiausias)

#### 2. Išskirtys (Outliers)

**Juodas apvadas** (⚫) → Vidinės išskirtys

- Nustatytos pagal 1.5×IQR taisyklę
- Šiek tiek nutolę nuo pagrindinės masės
- Gali būti "normalios" kraštinės reikšmės

**Raudonas apvadas** (🔴) → Išorinės išskirtys

- Nustatytos pagal 3×IQR taisyklę
- Labai nutolę nuo pagrindinės masės
- Tikėtina, kad tai ekstremalūs atvejai

#### 3. Pozicija Erdvėje

- **Artimi taškai** → Panašūs objektai (panašūs požymiai)
- **Tolie taškai** → Skirtingi objektai
- **Klasteris** → Grupė panašių objektų
- **Tuščia erdvė** → Nėra objektų su tokiomis savybėmis

### Kas Reiškia Gera Vizualizacija?

✅ **Gerai:**

```
Config 2, Euclidean, Normalized
Silhouette: 0.623, Purity: 0.847

Vizualizacija:
- Trys aiškiai atskirti klasteriai (rožinė, mėlyna, žalia)
- Minimali tarpų persidengia
- Išskirtys ant klasterių pakraščių
```

❌ **Blogai:**

```
Config 4, Cosine, Raw
Silhouette: 0.142, Purity: 0.523

Vizualizacija:
- Visos spalvos sumaišytos
- Nėra aiškių klasterių
- Išskirtys visur pasklidę
```

---

## Metrikų Interpretacija

### Silhouette Score

**Formulė idėja:**  
Matuoja, kiek objektas artimas savo klasteriui, palyginti su kitais klasteriais.

**Interpretacija:**

| Reikšmė       | Vertinimas      | Kas Tai Reiškia                           |
| ------------- | --------------- | ----------------------------------------- |
| **0.71-1.0**  | 🌟 Puiku        | Klasteriai labai aiškiai atskirti         |
| **0.51-0.70** | ✅ Gerai        | Klasteriai aiškiai atskirti               |
| **0.26-0.50** | ⚠️ Vidutiniškai | Klasteriai atskirti, bet persidengimų yra |
| **0.00-0.25** | ❌ Blogai       | Klasteriai labai persidengę               |
| **< 0**       | 💀 Labai blogai | Objektai neteisingai priskirti            |

**Praktinis pavyzdys:**

```
Silhouette = 0.65
→ Geras atskyrimas. Nutukimo tipus galima atskirti pagal elgsenos požymius.
→ Tinka tolimesnei analizei/klasifikavimui.

Silhouette = 0.18
→ Prastas atskyrimas. Nutukimo tipai panašūs pagal šiuos požymius.
→ Reikia papildomų požymių arba kito metodo.
```

### Purity Score

**Formulė idėja:**  
Kiek % kiekvieno klasterio dominuoja viena klasė (nutukimo tipas)?

**Interpretacija:**

| Reikšmė       | Vertinimas      | Kas Tai Reiškia                               |
| ------------- | --------------- | --------------------------------------------- |
| **0.90-1.0**  | 🌟 Puiku        | Klasteriai beveik 100% gryni (vienos klasės)  |
| **0.75-0.89** | ✅ Gerai        | Klasteriai dažniausiai vienos klasės          |
| **0.60-0.74** | ⚠️ Vidutiniškai | Klasteriai mišrūs, bet yra dominuojanti klasė |
| **< 0.60**    | ❌ Blogai       | Klasteriai chaotiški, klasės sumaišytos       |

**Praktinis pavyzdys:**

```
Purity = 0.84
→ 84% klasterio objektų priklauso vienai klasei.
→ Klasteriai gerai atitinka tikrąsias klases (nutukimo tipus).

Purity = 0.55
→ Tik 55% dominuoja – klasteriai mišrūs.
→ UMAP nepavyko suskirstyti pagal klases.
```

### Kombinuota Interpretacija

| Silhouette  | Purity      | Interpretacija                                                            |
| ----------- | ----------- | ------------------------------------------------------------------------- |
| **Aukštas** | **Aukštas** | 🏆 **IDEALUS** – Klasteriai aiškūs IR atitinka klases                     |
| **Aukštas** | **Žemas**   | 🤔 Klasteriai aiškūs, bet ne pagal nutukimo tipus (gal kitas grupavimas?) |
| **Žemas**   | **Aukštas** | 🧩 Klasteriai persidengę, bet klasės vis tiek sugrupuotos                 |
| **Žemas**   | **Žemas**   | ❌ **BLOGIAUSIAS** – Nei klasterių, nei klasių atitikimo                  |

---

## Konfigūracijų Palyginimas

### Konfigūracija 1: Detali Lokali (n=15, d=0.1)

**Charakteristika:**

- Mažas n_neighbors → akcentuoja lokalius santykius
- Mažas min_dist → tankūs, kompaktiški klasteriai

**Kada Naudoti:**
✅ Išskirčių identifikavimui  
✅ Smulkių subgrupių tyrimui  
✅ Lokalioms struktūroms

**Kada Nenaudoti:**
❌ Triukšmingiems duomenims (sukuria daug smulkių klasterių)  
❌ Globaliam klasių atskyrimui

**Tipiniai Rezultatai:**

- Silhouette: 0.45-0.65
- Purity: 0.70-0.85
- Daug mažų klasterių, išskirtys ryškios

**Pavyzdys:**

```
Config 1, Euclidean, Normalized
Silhouette: 0.58, Purity: 0.79

Matome:
- 5-7 mažus klasterius (ne tik 3 tipus)
- Kiekvienas tipas suskyla į subgrupes
- Išskirtys labai ryškios (raudonos / juodos)

Išvada:
→ Nutukimo tipuose yra subgrupės
→ Išskirtys – ekstremalūs atvejai
```

### Konfigūracija 2: Subalansuota (n=25, d=0.2)

**Charakteristika:**

- Vidutinis n_neighbors → balansas
- Vidutinis min_dist → nei per tankus, nei per sklaidus

**Kada Naudoti:**
✅ **REKOMENDUOJAMA PRADĖTI NUO ŠIOS**  
✅ Bendrai vizualizacijai  
✅ Pristatymams  
✅ Kai nežinai, ko tikėtis

**Kada Nenaudoti:**
❌ Kai reikia labai detalaus ar labai globalaus vaizdo

**Tipiniai Rezultatai:**

- Silhouette: 0.50-0.70
- Purity: 0.75-0.90
- 3-5 aiškūs klasteriai

**Pavyzdys:**

```
Config 2, Manhattan, Normalized
Silhouette: 0.64, Purity: 0.86

Matome:
- 3 pagrindiniai klasteriai (atitinka 3 tipus)
- Šiek tiek persidengimų tarp tipų 4 ir 5
- Išskirtys ant klasterių pakraščių

Išvada:
→ Nutukimo tipus galima atskirti
→ Tarp lengvesnio ir vidutinio nutukimo nėra griežtos ribos
```

### Konfigūracija 3: Globali (n=40, d=0.3)

**Charakteristika:**

- Didelis n_neighbors → globalios struktūros
- Didesnis min_dist → sklidesni klasteriai

**Kada Naudoti:**
✅ Klasių atskyrimo vertinimui  
✅ Triukšmo mažinimui  
✅ Globalių tendencijų tyrimui

**Kada Nenaudoti:**
❌ Smulkioms detalėms  
❌ Išskirčių tyrimui

**Tipiniai Rezultatai:**

- Silhouette: 0.40-0.60
- Purity: 0.80-0.95
- 2-3 dideli klasteriai

**Pavyzdys:**

```
Config 3, Euclidean, Raw
Silhouette: 0.48, Purity: 0.91

Matome:
- 3 aiškiai atskirti klasteriai (mažai persidengimų)
- Kiekvienas klasteris ~95% vienos spalvos
- Išskirtys retai (tik labai ekstremalūs)

Išvada:
→ Nutukimo tipai LABAI skirtingi globaliai
→ Nepaisant lokalių panašumų, bendras skirtumas aiškus
```

### Konfigūracija 4: Labai Globali (n=45, d=0.9)

**Charakteristika:**

- Labai didelis n_neighbors → labai globali struktūra
- Labai didelis min_dist → labai sklaidūs klasteriai

**Kada Naudoti:**
✅ Labai aukšto lygio apžvalgai  
✅ Kai reikia "big picture"

**Kada Nenaudoti:**
❌ **BEVEIK VISADA** – per daug prarandama informacijos  
❌ Detaliai analizei

**Tipiniai Rezultatai:**

- Silhouette: 0.20-0.40
- Purity: 0.70-0.85
- 1-2 dideli klasteriai (viską sujungia)

**Pavyzdys:**

```
Config 4, Cosine, Normalized
Silhouette: 0.28, Purity: 0.76

Matome:
- Viską sujungė į 1-2 dideles grupes
- Spalvos sumaišytos
- Prarastos visos detalės

Išvada:
→ Per daug agregavimo
→ Nenaudotina praktiškai
```

---

## Normalizacijos Efektas

### Raw (Nenormalizuoti) Duomenys

**Savybės:**

- Išlaiko originalias skales (pvz., Gender 0-1, FCVC 1-3)
- Didesni požymiai dominuoja

**Kada Gerai:**

- Jei originalios skalės informatyvios
- Jei požymiai natūraliai panašių diapazonų

**Pavyzdys:**

```
Config 2, Euclidean, Raw
Silhouette: 0.42

Vidinė analizė:
- Požymis "Gender" (0-1) beveik neturi įtakos
- Požymis "FCVC" (1-3) dominuoja
- Požymis "CH2O" (1-3) taip pat svarbus

Išvada:
→ Rezultatas pagrįstas vien maistavimo įpročiais
→ Demografija ignoruojama
```

### Normalized (Min-Max) Duomenys

**Savybės:**

- Visi požymiai [0, 1] diapazone
- Visi požymiai vienodai svarbūs

**Kada Gerai:**

- Kai visi požymiai svarbūs
- Kai skalės labai skirtingos

**Pavyzdys:**

```
Config 2, Euclidean, Normalized
Silhouette: 0.64

Vidinė analizė:
- Visi požymiai vienodai svarbūs
- Gender+FCVC+SMOKE = vienodai lemia atstumą
- Rezultatas holistiškas

Išvada:
→ Rezultatas atsižvelgia į visus veiksnius
→ Geresnė klasterių kokybė (+0.22 Silhouette)
```

### Kaip Nuspręsti?

**Palyginkite vidutinius rezultatus:**

```
Vidutinis Silhouette (Raw): 0.38
Vidutinis Silhouette (Norm): 0.61

Skirtumas: +0.23 → NORMALIZACIJA PAGERINA

Rekomendacija:
✅ Naudokite normalizuotus duomenis
```

```
Vidutinis Silhouette (Raw): 0.58
Vidutinis Silhouette (Norm): 0.49

Skirtumas: -0.09 → NORMALIZACIJA PABLOGINA

Rekomendacija:
✅ Naudokite nenormalizuotus duomenis
   (originalios skalės yra informatyvios)
```

---

## Išskirčių Analizė

### Kas yra Išskirtys?

**Vidinės išskirtys (1.5×IQR):**

- Nutolę nuo pagrindinės masės, bet ne labai
- ~5-10% duomenų
- Gali būti normalios kraštinės reikšmės

**Išorinės išskirtys (3×IQR):**

- Labai nutolę
- ~1-3% duomenų
- Tikėtina, kad ekstremalūs atvejai arba klaidos

### Kaip Interpretuoti Išskirtis Vizualizacijoje?

#### Scenarijus 1: Išskirtys Ant Klasterių Pakraščių

```
Visualization:
- Pagrindiniai klasteriai kompaktiški
- Juodi/raudoni apvadai ant pakraščių
- Išskirtys "tęsia" klasterio kryptį

Interpretacija:
→ Išskirtys = ekstremalūs to paties tipo atvejai
→ Pavyzdys: Labai sunkūs Obesity Type III atvejai
→ Veiksmas: Normalios reikšmės, palikti duomenyse
```

#### Scenarijus 2: Išskirtys Tarpuose

```
Visualization:
- Pagrindiniai klasteriai atskirti
- Juodi/raudoni apvadai tuščiose erdvėse

Interpretacija:
→ Išskirtys = pereinamieji atvejai (tarp tipų)
→ Pavyzdys: Asmuo pereiną iš Type I į Type II
→ Veiksmas: Svarbi informacija, tikrai palikti
```

#### Scenarijus 3: Išskirtys Visur Chaotiškai

```
Visualization:
- Klasteriai neaiškūs
- Juodi/raudoni apvadai visur

Interpretacija:
→ Duomenys labai heterogeniški ARBA
→ Požymiai netinkami klasifikacijai
→ Veiksmas: Tikrinti duomenų kokybę
```

### Išskirčių Statistika

**Pavyzdys:**

```
Vidinės išskirtys: 68 (7.0%)
Išorinės išskirtys: 23 (2.4%)
Viso išskirčių: 91 (9.4%)

Pasiskirstymas pagal klases:
- Tipas 4: 12 išskirčių (15% visos klasės)
- Tipas 5: 31 išskirtis (9% visos klasės)
- Tipas 6: 48 išskirtys (12% visos klasės)

Interpretacija:
→ ~10% išskirčių – normalus kiekis
→ Visos klasės turi panašų išskirčių %
→ Nėra "problemingesnės" klasės
```

---

## Praktiniai Pavyzdžiai

### Pavyzdys 1: Geriausia Konfigūracija

**Situacija:**

```
Config 2, Manhattan, Normalized
Silhouette: 0.712
Purity: 0.893
Išskirtys: 8.7%
```

**Vizualizacija:**

- 3 aiškiai atskirti klasteriai
- Rožinė (Tipas 4) kairėje
- Mėlyna (Tipas 5) viduryje
- Žalia (Tipas 6) dešinėje
- Minimalūs persidengimų tarp mėlynos ir rožinės
- Išskirtys ant klasterių pakraščių

**Išvados:**

1. ✅ Nutukimo tipus GALIMA atskirti pagal elgsenos požymius
2. ✅ Lengviausias (4) ir sunkiausias (6) tipai labai skirtingi
3. ⚠️ Lengviausias (4) ir vidutinis (5) turi panašumų
4. ✅ Normalizacija pagerina atskyrumą
5. ✅ Manhattan metrika geriau nei Euclidean šiems duomenims

**Rekomendacijos:**

- Naudoti šią konfigūraciją tolimesnei analizei
- Klasifikavimo modeliams turėtų gerai sekti
- Išskirti Tipą 6 nuo kitų bus lengva, Tipus 4 vs 5 – sunkiau

### Pavyzdys 2: Prasta Konfigūracija

**Situacija:**

```
Config 4, Cosine, Raw
Silhouette: 0.183
Purity: 0.541
Išskirtys: 12.3%
```

**Vizualizacija:**

- Visos spalvos sumaišytos
- Nėra aiškių klasterių
- Išskirtys visur chaotiškai

**Išvados:**

1. ❌ Ši konfigūracija netinkama
2. ❌ Cosine metrika netinkama šiems duomenims
3. ❌ Nenormalizuoti duomenys blogina rezultatus
4. ❌ n_neighbors=45 + min_dist=0.9 per daug agregavo

**Rekomendacijos:**

- NENAUDOTI šios konfigūracijos
- Pereiti prie Config 2 ar 3
- Bandyti Euclidean ar Manhattan
- Naudoti normalizuotus duomenis

### Pavyzdys 3: Normalizacijos Palyginimas

**Situacija:**

```
Config 2, Euclidean:
  Raw:        Silhouette 0.412, Purity 0.758
  Normalized: Silhouette 0.641, Purity 0.871

Skirtumas: +0.229 Silhouette, +0.113 Purity
```

**Išvada:**
✅ **NORMALIZACIJA STIPRIAI PAGERINA**

**Priežastys:**

- Raw: FCVC (1-3) dominuoja, Gender (0-1) ignoruojamas
- Normalized: Visi požymiai vienodai svarbūs
- Rezultatas: Holistiškesnis požiūris → geresnis atskyrimas

**Rekomendacija:**

- VISUOMET naudoti normalizuotus duomenis šiam rinkiniui
- Galima bandyti ir kitas normalizacijas (Z-score)

---

## Dažniausios Klaidos

### Klaida 1: "Visos spalvos sumaišytos – metodas netinka"

**Problema:** Nėra aiškių klasterių pagal spalvas

**Galimos priežastys:**

1. ❌ Netinkama konfigūracija (pvz., Config 4)
2. ❌ Netinkama metrika (pvz., Cosine vietoj Euclidean)
3. ❌ Nenormalizuoti duomenys
4. ⚠️ Požymiai nepakankami klasių atskyrimui

**Sprendimas:**

1. Išbandyti Config 2 su Euclidean ir Normalized
2. Jei vis tiek blogai → pridėti daugiau požymių
3. Palyginti su kitais metodais (t-SNE, PCA)

### Klaida 2: "Per daug išskirčių (>20%)"

**Problema:** Daugiau nei 20% duomenų pažymėta kaip išskirtys

**Galimos priežastys:**

1. ⚠️ IQR taisyklė per griežta šiems duomenims
2. ⚠️ Duomenys labai heterogeniški
3. ❌ Duomenų kokybės problemos

**Sprendimas:**

1. Patikrinti išskirčių reikšmes (ar tikrai ekstremalios?)
2. Svarstyti kitas išskirčių nustatymo metodus
3. Tikrinti duomenų įvedimo klaidas

### Klaida 3: "Normalizacija pablogino rezultatus"

**Problema:** Normalized Silhouette mažesnis nei Raw

**Galimos priežastys:**

1. ✅ Originalios skalės yra informatyvios
2. ✅ Kai kurie požymiai TURI būti svarbesni

**Sprendimas:**

1. Naudoti Raw duomenis
2. Arba išbandyti feature weighting
3. Arba svarstomąją normalizaciją

### Klaida 4: "Config 1 ir Config 3 labai skirtingi rezultatai"

**Problema:** Silhouette skiriasi >0.3 tarp konfigūracijų

**Galimos priežastys:**

1. ✅ NORMALUS! Skirtingos konfigūracijos rodo skirtingus aspektus
2. ⚠️ Config 1 jautrus triukšmui
3. ⚠️ Config 3 slopina detales

**Sprendimas:**

1. Naudoti Config 2 kaip kompromisą
2. Remtis abiejų interpretacija
3. Stabilumo analizė (skirtingi random_state)

---

## Santrauka: Kaip Priimti Sprendimus

### Žingsnis 1: Peržiūrėti TOP 5 Konfigūracijas

```python
# Iš umap_results_summary.csv
top5 = df.nlargest(5, 'silhouette')
```

### Žingsnis 2: Patikrinti Abiejų Metrikų Balansą

```
Ideal: Silhouette > 0.5 IR Purity > 0.8
```

### Žingsnis 3: Palyginti Raw vs Normalized

```
Jei Normalized gerina >0.1 → naudoti Normalized
```

### Žingsnis 4: Vizualiai Patikrinti TOP Konfigūraciją

```
- Ar klasteriai aiškūs?
- Ar spalvos atskirtos?
- Ar išskirtys logiškose vietose?
```

### Žingsnis 5: Nuspręsti

```
✅ Geriausia konfigūracija = aukščiausi Silhouette+Purity + aiški vizualizacija
```

---

**Sėkmės analizuojant! 🚀**
