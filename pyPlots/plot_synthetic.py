import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# Style
sns.set(style="whitegrid")

def style_plot(title, xlabel, ylabel):
    #plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=24)
    plt.ylabel(ylabel, fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(title="Algorithm", fontsize=18, title_fontsize=18)
    plt.tight_layout()

N_values = [32, 64, 128]
selected_es = [0.01, 0.5, 1, 1.5, 2]

palette = {
    "FIT": "blue",
    "IFI": "red",
    "IESJ": "orange",
    "BF": "purple",
    "RNLJ": "green"
}
markers = {
    "FIT": 'v',
    "IFI": 'D',
    "IESJ": 's',
    "BF": '*',
    "RNLJ": 'o'
}

# Plotting
for N in N_values:
    csv_files = glob.glob(f"Results/UsefulE/*{N}.csv")
    df_list = []

    for file in csv_files:
        df = pd.read_csv(file, header=None)
        df.columns = ["Algorithm", "n", "e", "Runtime", "MemoryMB", "RunID"]
        df_list.append(df)

    # Combine all results
    results = pd.concat(df_list, ignore_index=True)

    # Convert data types
    results["n"] = results["n"].astype(int)
    results["e"] = results["e"].astype(float)

    selected_es = results["e"].unique()

    # Filter for selected e values only
    results = results[results["e"].isin(selected_es)]

    # Aggregate results (mean over runs)
    agg = results.groupby(["Algorithm", "n", "e"]).agg({
        "Runtime": "mean",
        "MemoryMB": "mean"
    }).reset_index()

    num_algorithms = agg["Algorithm"].nunique()
    #palette = sns.color_palette("husl", num_algorithms)

    unique_es = sorted(agg["e"].unique())
    unique_ns = sorted(agg["n"].unique())

    # Ensure output directories exist
    os.makedirs("plots/runtime", exist_ok=True)
    os.makedirs("plots/memory", exist_ok=True)

    # ===== Runtime vs n (linear and log scale) =====
    for e_val in unique_es:
        plot_data = agg[agg["e"] == e_val]
        subset = plot_data.copy()
        subset["n"] = subset["n"]/1000

        # Linear scale
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=subset, x="n", y="Runtime", style="Algorithm", hue="Algorithm", markers=markers,dashes=False, markersize=16, palette=palette, linewidth=4)
        style_plot(f"Runtime vs n (e={e_val})", "Length (n)x1000", "Runtime (s)")
        plt.savefig(f"plots/runtime/runtime_vs_n_e{e_val}_N{N}.png")
        plt.close()

        # Log scale
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=subset, x="n", y="Runtime", hue="Algorithm",style="Algorithm", markers=markers, dashes=False, markersize=16, palette=palette, linewidth=4)
        plt.yscale("log")
        style_plot(f"Runtime vs n (log scale) (e={e_val})", "Length (n) x 1000", "Runtime (log seconds)")
        plt.savefig(f"plots/runtime/runtime_vs_n_log_e{e_val}_N{N}.png")
        plt.close()

    # ===== Runtime vs e (linear and log scale) =====
    for n_val in unique_ns:
        subset = agg[agg["n"] == n_val]

        # Linear scale
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=subset, x="e", y="Runtime", hue="Algorithm",style="Algorithm", markers=markers, dashes=False, markersize=16, palette=palette, linewidth=4)
        style_plot(f"Runtime vs e (n={n_val})", r"$\epsilon$", "Runtime (s)")
        plt.savefig(f"plots/runtime/runtime_vs_e_n{n_val}_N{N}.png")
        plt.close()

        # Log scale
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=subset, x="e", y="Runtime", hue="Algorithm",style="Algorithm", markers=markers, dashes=False, markersize=16, palette=palette, linewidth=4)
        plt.yscale("log")
        style_plot(f"Runtime vs e (log scale) (n={n_val})", r"$\epsilon$", "Runtime (log seconds)")
        plt.savefig(f"plots/runtime/runtime_vs_e_log_n{n_val}_N{N}.png")
        plt.close()

    # ===== Memory Usage vs n for each e =====
    for e_val in unique_es:
        plot_data = agg[agg["e"] == e_val]
        subset = plot_data.copy()
        subset["n"] = subset["n"]/1000
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=subset, x="n", y="MemoryMB", hue="Algorithm", style="Algorithm", markers=markers, dashes=False, markersize=16, palette=palette, linewidth=4)
        style_plot(f"Memory Usage vs n (e={e_val})", "Length (n) x1000", "Memory Usage (MB)")
        plt.savefig(f"plots/memory/memory_vs_n_e{e_val}_N{N}.png")
        plt.close()

    # ===== Memory Usage vs e for each n =====
    for n_val in unique_ns:
        subset = agg[agg["n"] == n_val]
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=subset, x="e", y="MemoryMB", hue="Algorithm", style="Algorithm", markers=markers, markersize=16, dashes=False, palette=palette, linewidth=4)
        style_plot(f"Memory Usage vs e (n={n_val})", r"$\epsilon$", "Memory Usage (MB)")
        plt.savefig(f"plots/memory/memory_vs_e_n{n_val}_N{N}.png")
        plt.close()