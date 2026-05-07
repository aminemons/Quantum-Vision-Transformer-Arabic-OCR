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
    
    clean_df = df[df["Condition"] == "Clean"]
    noisy_df = df[df["Condition"] == "PGD_Adversarial"]
    
    # 1. Clean Accuracy Bar Plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=clean_df, x="Model", y="Accuracy", palette="viridis")
    plt.title("Clean Accuracy Comparison on 115 Classes (HMBD-v1)", fontsize=14, pad=15)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xlabel("Architecture", fontsize=12)
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.3f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha='center', va='center', xytext=(0, 9), textcoords='offset points')
    plt.savefig("accuracy_clean_115.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Adversarial Robustness (PGD) Bar Plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=noisy_df, x="Model", y="Accuracy", palette="magma")
    plt.title("Adversarial Robustness (PGD Attack) on 115 Classes", fontsize=14, pad=15)
    plt.ylabel("Accuracy Under Attack", fontsize=12)
    plt.xlabel("Architecture", fontsize=12)
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.3f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha='center', va='center', xytext=(0, 9), textcoords='offset points')
    plt.savefig("accuracy_pgd_115.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Efficiency Bar Plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=clean_df, x="Model", y="AccuracyPer1000Params", palette="coolwarm")
    plt.title("Parameter Efficiency (Clean Acc / 1k Params)", fontsize=14, pad=15)
    plt.ylabel("Efficiency Score", fontsize=12)
    plt.xlabel("Architecture", fontsize=12)
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.3f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha='center', va='center', xytext=(0, 9), textcoords='offset points')
    plt.savefig("efficiency_barplot_115.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print("Plots generated successfully: accuracy_clean_115.png, accuracy_pgd_115.png, efficiency_barplot_115.png")

if __name__ == "__main__":
    generate_plots()
