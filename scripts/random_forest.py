import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from evaluation import holdout_evaluate, crossval_evaluate


def main():
    df = pd.read_csv("./data/classification_data.csv")

    target_col = "NObeyesdad"

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        )
    }

    # Hold-out
    holdout_results, _ = holdout_evaluate(models, X, y, test_size=0.2, random_state=42)
    print("=== HOLD-OUT (Random Forest) ===")
    res = holdout_results["random_forest"]
    print(f"Accuracy: {res['accuracy']:.4f}")
    print(res["classification_report"])

    # Cross-val
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
