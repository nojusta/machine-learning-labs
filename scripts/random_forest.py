import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from evaluation import holdout_evaluate, crossval_evaluate


def plot_2d_decision_surface(X, y, feature_names, model):
    """
    Paprastas 2D sprendimo ribos (decision boundary) vaizdavimas.
    Naudoja tik 2 požymius iš X: feature_names[0] ir feature_names[1]
    """

    f1, f2 = feature_names

    # Paimam tik dvi kolonas
    X_2d = X[[f1, f2]].values
    y_vals = y.values if hasattr(y, "values") else y

    # Apmokome modelį tik ant šių dviejų požymių
    model.fit(X_2d, y_vals)

    # Sukuriam tinklelį (meshgrid) sprendimo ribai nubrėžti
    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200),
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))

    # Fonas – modelio prognozuotos klasės
    plt.contourf(xx, yy, Z, alpha=0.3)

    # Tikrieji duomenų taškai
    scatter = plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=y_vals,
        edgecolor="k",
        alpha=0.8,
    )

    plt.xlabel(f1)
    plt.ylabel(f2)
    plt.title("Random Forest klasifikavimo vaizdas 2D plokštumoje")

    # Legenda pagal klases
    handles, _ = scatter.legend_elements()
    class_labels = np.unique(y_vals)
    plt.legend(handles, [str(c) for c in class_labels], title="Klasė")

    plt.tight_layout()
    plt.show()


def main():
    df = pd.read_csv("./data/classification_data_tsne.csv")

    target_col = "NObeyesdad"

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        )
    }

    # ====== HOLD-OUT ======
    holdout_results, _ = holdout_evaluate(
        models, X, y, test_size=0.2, random_state=42
    )

    print("=== HOLD-OUT (Random Forest) ===")
    res = holdout_results["random_forest"]
    print(f"Accuracy: {res['accuracy']:.4f}")
    print(res["classification_report"])

    # ====== CROSS-VAL ======
    cv_results = crossval_evaluate(
        models,
        X,
        y,
        cv=5,
        scoring={"accuracy": "accuracy", "f1_weighted": "f1_weighted"},
    )

    print("\n=== CROSS-VAL (Random Forest) ===")
    for metric, value in cv_results["random_forest"].items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()