import matplotlib.pyplot as plt

# Klasifikatorių pavadinimai
models = ["Decision Tree", "k-NN", "Random Forest"]

# Kiekvienoje eilutėje: [Hold-out, 5-fold CV]
accuracy = [
    [0.737, 0.831],  # DT
    [0.823, 0.866],  # k-NN
    [0.853, 0.899],  # RF
]

precision = [
    [0.739, 0.830],  # DT
    [0.776, 0.925],  # k-NN
    [0.853, 0.901],  # RF
]

recall = [
    [0.737, 0.833],  # DT
    [0.938, 0.833],  # k-NN
    [0.853, 0.899],  # RF
]

f1 = [
    [0.736, 0.829],  # DT
    [0.849, 0.874],  # k-NN
    [0.853, 0.899],  # RF
]

metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
}

for metric_name, values in metrics.items():
    plt.figure(figsize=(7, 4))
    plt.boxplot(
        values,
        labels=models,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", color="blue"),
        medianprops=dict(color="black", linewidth=1.5),
    )
    plt.ylabel(metric_name)
    plt.title(f"Klasifikatorių {metric_name} rezultatai (Hold-out ir 5-fold CV)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{metric_name.lower().replace(' ', '_')}_boxplot.png", dpi=300)
    plt.show()