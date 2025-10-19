@echo off
REM ========================================
REM UMAP Pilna Analize - Automatinis Paleidimas
REM ========================================

echo.
echo ============================================================
echo  UMAP DIMENSIJU MAZINIMO ANALIZE
echo ============================================================
echo.
echo Sitas skriptas paleis:
echo   1. Pilna UMAP analize (24 vizualizacijos)
echo   2. Palyginimo grafiku generavima
echo.
echo Prognozuojama trukme: 10-20 minuciu
echo.
pause

REM Patikriname Python
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo KLAIDA: Python nerastas!
    echo Prasome idiegti Python is https://www.python.org/
    pause
    exit /b 1
)

echo.
echo [1/3] Tikriname priklausomybes...
python -c "import pandas, numpy, matplotlib, umap, sklearn" 2>nul
if errorlevel 1 (
    echo.
    echo Idiegiame trukstamas bibliotekas...
    pip install pandas numpy matplotlib seaborn scikit-learn umap-learn
    if errorlevel 1 (
        echo.
        echo KLAIDA: Nepavyko idiegti biblioteku!
        pause
        exit /b 1
    )
)
echo OK - Visos priklausomybes idiegtos

echo.
echo [2/3] Vykdoma pilna UMAP analize...
echo (tai gali uztrukti 10-20 minuciu, prasome kantrybės)
echo.
python scripts/umap_analysis_full.py
if errorlevel 1 (
    echo.
    echo KLAIDA analizės metu!
    pause
    exit /b 1
)

echo.
echo [3/3] Generuojami palyginimo grafikai...
echo.
python scripts/umap_comparison_plots.py
if errorlevel 1 (
    echo.
    echo KLAIDA grafiku generavime!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  ANALIZE BAIGTA SEKMINGAI!
echo ============================================================
echo.
echo Rezultatai issaugoti:
echo   - outputs/umap/ - visos vizualizacijos
echo   - outputs/umap/UMAP_ANALIZE_ATASKAITA.md - ataskaita
echo   - outputs/umap/umap_results_summary.csv - metrikos
echo.
echo Norėdami peržiūrėti rezultatus, atidarykite:
echo   outputs\umap\UMAP_ANALIZE_ATASKAITA.md
echo.
pause

REM Atidarome ataskaita (jei yra)
if exist "outputs\umap\UMAP_ANALIZE_ATASKAITA.md" (
    echo.
    echo Ar norite atidaryti ataskaita dabar? (T/N)
    set /p open_report=
    if /i "%open_report%"=="T" (
        start "" "outputs\umap\UMAP_ANALIZE_ATASKAITA.md"
    )
)

REM Atidarome output kataloga
if exist "outputs\umap\" (
    echo.
    echo Ar norite atidaryti rezultatu kataloga? (T/N)
    set /p open_folder=
    if /i "%open_folder%"=="T" (
        start "" "outputs\umap\"
    )
)

echo.
echo Acui!
pause
