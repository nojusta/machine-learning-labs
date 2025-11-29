import pandas as pd
from sklearn.manifold import TSNE

def main():
    # 1. Užkraunam klasifikavimo aibę (6 požymiai + klasė)
    df = pd.read_csv("./data/classification_data.csv")

    feature_cols = ["FCVC", "FAF", "CH2O", "NCP", "TUE", "MTRANS"]
    target_col = "NObeyesdad"

    print("Kraunami požymiai klasifikavimui:")
    print(feature_cols)

    # 2. X t-SNE skaičiavimui (eilės tvarka nekeičiama)
    X_for_tsne = df[feature_cols].values
    y = df[target_col].values

    print("\nKuriama t-SNE 2D projekcija...")
    tsne = TSNE(
        n_components=2,
        perplexity=40,
        max_iter=600,
        random_state=42,
    )
    X_tsne = tsne.fit_transform(X_for_tsne)

    print(f"t-SNE projekcijos forma: {X_tsne.shape}")

    # 3. Pridedame t-SNE koordinatės prie originalaus DF
    df["tsne_1"] = X_tsne[:, 0]
    df["tsne_2"] = X_tsne[:, 1]

    # 4. Sukuriame galutinį failą klasterizacijai:
    #    tik t-SNE 2D + klasė

    df_final = pd.DataFrame({
        "tsne_1": df["tsne_1"],
        "tsne_2": df["tsne_2"],
        "NObeyesdad": df[target_col]
    })

    out_cluster = "./data/classification_data_tsne.csv"
    df_final.to_csv(out_cluster, index=False)

    print(f"\nKlasterizacijai skirtas failas išsaugotas į: {out_cluster}")
    print("Struktūra:")
    print(df_final.head())

if __name__ == "__main__":
    main()