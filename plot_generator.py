import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_plots(csv_path="results_comparison.csv"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run train.py first.")
        return

    df = pd.read_csv(csv_path)
    
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(10, 6))
    clean_df = df[df["Condition"] == "Clean"]
    sns.lineplot(data=clean_df, x="Classes", y="Accuracy", hue="Model", marker="o", linewidth=2.5)
    plt.title("Accuracy vs. Class Count (Clean Context)")
    plt.ylabel("Accuracy")
    plt.xlabel("Number of Classes")
    plt.xticks([28, 115])
    plt.ylim(0, 1.1)
    plt.legend(title="Architecture")
    plt.savefig("accuracy_vs_class_count_clean.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    noisy_df = df[df["Condition"] == "Noisy"]
    sns.lineplot(data=noisy_df, x="Classes", y="Accuracy", hue="Model", marker="s", linewidth=2.5)
    plt.title("Robustness: Accuracy vs. Class Count (Noisy Stress Test)")
    plt.ylabel("Accuracy")
    plt.xlabel("Number of Classes")
    plt.xticks([28, 115])
    plt.ylim(0, 1.1)
    plt.legend(title="Architecture")
    plt.savefig("accuracy_vs_class_count_noisy.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=clean_df, x="Model", y="AccuracyPer1000Params", hue="Classes")
    plt.title("Efficiency: Accuracy per 1000 Parameters")
    plt.ylabel("Accuracy per 1000 Params")
    plt.xlabel("Model")
    plt.savefig("efficiency_barplot.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print("Plots generated successfully: accuracy_vs_class_count_clean.png, accuracy_vs_class_count_noisy.png, efficiency_barplot.png")

if __name__ == "__main__":
    generate_plots()
