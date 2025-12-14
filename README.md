# Machine Learning Course Labs

Repository for ML course lab work. Each lab lives on its own branch:

- **LD1 (financial dataset)**: data cleaning, missing-value handling, normalization, basic descriptive stats to prep the dataset.
- **LD2 (obesity classification, 2D viz)**: PCA, MDS, t-SNE, UMAP on the obesity dataset; plots and brief comparisons.
- **LD3 (clustering)**: K-Means, AGNES (hierarchical), DBSCAN experiments with metrics/plots.
- **LD4 (classification models)**: KNN, Decision Tree, Random Forest on the obesity dataset with validation/test splits, ROC/AUC, confusion matrices, and error analysis.

## How we work
1) Create a feature branch: `name/ldX-topic`
2) Add code/notebooks under the appropriate lab folder
3) Commit: `git add . && git commit -m "LDX: short description"`
4) Push and open a PR into the matching lab branch

## Guidelines
- Test before PR; include plots/results where relevant
- Keep large raw data/model artifacts out of git (use `.gitignore`)
- Python 3.x; install lab-specific requirements as needed
