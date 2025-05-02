import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob

def get_configure_plot_style():
    # ======================
    # CONSTANTS AND SETTINGS
    # ======================
    return  {
        'IFI': '#3366cc',  # Blue
        'FIT': '#dc3912',  # Red
        'BF': '#ff9900',   # Orange
        'IESJ': '#109618', # Green
        'RNLJ': '#990099'  # Purple
        },{
            'figsize': (12, 10),
            'title_fontsize': 14,
            'label_fontsize': 22,
            'tick_fontsize': 20,  # Specific font size for ticks
            'legend_fontsize': 16,
            'grid_alpha': 0.2,
            'dpi': 300
        }
# ======================
# DATA LOADING AND PREP
# ======================
def load_and_prepare_data():
    files = glob.glob("Exp_results/Exp_edge/*.csv")  
    dfs = []
    
    for file in files:
        df = pd.read_csv(file, header=None)
        df.columns = ["algorithm", "n", "e", "runtime", "memory", "run_id"]
        algo_name = file.split("/")[-1]
        algo_name = algo_name.split("_")[0]
        df["algorithm"] = algo_name
        dfs.append(df)

    data = pd.concat(dfs).reset_index(drop=True)
    
    # Label edge cases
    data["edge_case"] = data.groupby("algorithm").cumcount() + 1
    data["edge_case"] = data["edge_case"].map({1: "Case 1", 2: "Case 2"})
    data["status"] = "ok"

    # Handle missing cases
    expected_cases = set(["Case 1", "Case 2"])
    for algo in data["algorithm"].unique():
        present_cases = set(data[data["algorithm"] == algo]["edge_case"])
        missing = expected_cases - present_cases
        for case in missing:
            data = pd.concat([
                data,
                pd.DataFrame([{
                    "algorithm": algo,
                    "n": None,
                    "e": None,
                    "runtime": None,
                    "memory": None,
                    "run_id": 0,
                    "edge_case": case,
                    "status": "timeout"
                }])
            ], ignore_index=True)

    # Prepare plotting columns
    data["runtime_plot"] = data["runtime"].fillna(20*3600)  # Timeout -> 20h
    data["runtime_plot"] = np.where(data["runtime_plot"] < 1e-3, 1e-3, data["runtime_plot"])
    
    # Convert memory from MB to KB and handle timeouts
    data["memory_kb"] = data["memory"] * 1024
    data["memory_plot"] = data["memory_kb"].fillna(1e7)  # 1 KB minimum for log scale
    
    return data, data[data["status"] == "timeout"]

# ======================
# PLOTTING FUNCTIONS
# ======================
def configure_plot_style():
    
    ALGORITHM_COLORS ,PLOT_SETTINGS = get_configure_plot_style()
    plt.style.use('seaborn')
    sns.set_palette(sns.color_palette(ALGORITHM_COLORS.values()))
    plt.rcParams.update({
        'font.size': PLOT_SETTINGS['label_fontsize'],
        'axes.titlesize': PLOT_SETTINGS['title_fontsize'],
        'axes.labelsize': PLOT_SETTINGS['label_fontsize'],
        'xtick.labelsize': PLOT_SETTINGS['tick_fontsize'],
        'ytick.labelsize': PLOT_SETTINGS['tick_fontsize'],
        'grid.alpha': PLOT_SETTINGS['grid_alpha']
    })

def plot_runtime(data, timeout_data):
    # ======================
    # CONSTANTS AND SETTINGS
    # ======================
    ALGORITHM_COLORS ,PLOT_SETTINGS = get_configure_plot_style()
    plt.figure(figsize=PLOT_SETTINGS['figsize'])
    ax = sns.barplot(
        data=data,
        x="edge_case",
        y="runtime_plot",
        hue="algorithm",
        palette=ALGORITHM_COLORS,
        log=True
    )

    # Value annotations
    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            if height >= 1e-3:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height * 1.05,
                    f'{height:.1f}',
                    ha='center',
                    va='bottom',
                    fontsize=PLOT_SETTINGS['legend_fontsize'],
                    color='white' if height > 0.3 * ax.get_ylim()[1] else 'black',
                    fontweight='bold'
                )

    # Timeout annotations
    for _, row in timeout_data.iterrows():
        xpos = ["Case 1", "Case 2"].index(row["edge_case"])
        offset = 0.25 * (list(ALGORITHM_COLORS.keys()).index(row["algorithm"]) - 1.5)
        ax.text(
            xpos + offset,
            ax.get_ylim()[1] * 0.05e-1,
            "Timeout (>20h)",
            color="black",
            ha="center",
            fontsize=PLOT_SETTINGS['legend_fontsize']-1,
            rotation=90,
            bbox=dict(facecolor="red", alpha=0.3, edgecolor="none", boxstyle="round,pad=0.2")
        )

    ax.set_xlabel("Edge Case", fontweight='bold')
    ax.set_ylabel("Runtime (s) [log scale]", fontweight='bold')
    
    # Explicit tick control
    ax.tick_params(axis='both', which='major', labelsize=PLOT_SETTINGS['tick_fontsize'])
    
    # Explicit tick control
    ax.tick_params(axis='both', which='major', labelsize=PLOT_SETTINGS['tick_fontsize'])
    
    # Move legend below the plot
    plt.legend(
        fontsize=PLOT_SETTINGS['legend_fontsize'],
        framealpha=0.7,
        bbox_to_anchor=(0.5, -0.2),  # Positions legend below center
        loc='upper center',           # Anchors the legend
        ncol=5,                       # Number of columns for legend items
        borderaxespad=0.5             # Padding around legend
    )
    
    # Adjust plot margins to accommodate legend
    plt.subplots_adjust(bottom=0.25)  # Increase bottom margin
    
    plt.title("Algorithm Runtime Usage Across Edge Cases", fontweight='bold', pad=20)
    plt.grid(True, alpha=PLOT_SETTINGS['grid_alpha'])
    plt.savefig("Plots/runtime_edge_cases_log.png", dpi=PLOT_SETTINGS['dpi'], bbox_inches="tight")
    plt.show()

def plot_memory(data, timeout_data):
    ALGORITHM_COLORS ,PLOT_SETTINGS = get_configure_plot_style()
    plt.figure(figsize=PLOT_SETTINGS['figsize'])
    ax = sns.barplot(
        data=data,
        x="edge_case",
        y="memory_plot",
        hue="algorithm",
        palette=ALGORITHM_COLORS,
        log=True
    )

    # Value annotations
    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            v = height/1024
            if height > 1:  # Skip values ≤1 KB
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height * 1.05,
                    f'{v:,.2f} MB',
                    ha='center',
                    va='bottom',
                    fontsize=PLOT_SETTINGS['legend_fontsize'],
                    color='white' if height > 0.3 * ax.get_ylim()[1] else 'black',
                    fontweight='bold'
                )

    # Timeout annotations
    if "memory" in timeout_data and timeout_data["memory"].isna().any():
        for _, row in timeout_data.iterrows():
            xpos = ["Case 1", "Case 2"].index(row["edge_case"])
            offset = 0.25 * (list(ALGORITHM_COLORS.keys()).index(row["algorithm"]) - 1.5)
            ax.text(
                xpos + offset,
                ax.get_ylim()[1] * 0.05e-1,
                "Timeout > 20h",
                color="black",
                ha="center",
                fontsize=PLOT_SETTINGS['legend_fontsize'],
                rotation=90,
                bbox=dict(facecolor="red", alpha=0.3, edgecolor="none", boxstyle="round,pad=0.2")
            )

    ax.set_xlabel("Edge Case", fontweight='bold')
    ax.set_ylabel("Memory Usage (KB) [log scale]", fontweight='bold')
    
    # Explicit tick control
    ax.tick_params(axis='both', which='major', labelsize=PLOT_SETTINGS['tick_fontsize'])
    
    # Move legend below the plot
    plt.legend(
        fontsize=PLOT_SETTINGS['legend_fontsize'],
        framealpha=0.7,
        bbox_to_anchor=(0.5, -0.2),  # Positions legend below center
        loc='upper center',           # Anchors the legend
        ncol=5,                       # Number of columns for legend items
        borderaxespad=0.5             # Padding around legend
    )
    
    # Adjust plot margins to accommodate legend
    plt.subplots_adjust(bottom=0.25)  # Increase bottom margin
    
    plt.title("Algorithm Memory Usage Across Edge Cases", fontweight='bold', pad=20)
    plt.grid(True, alpha=PLOT_SETTINGS['grid_alpha'])
    plt.savefig("Plots/memory_edge_cases_log_kb.png", dpi=PLOT_SETTINGS['dpi'], bbox_inches="tight")
    plt.show()
# ======================
# MAIN EXECUTION
# ======================
if __name__ == "__main__":
    data, timeout_data = load_and_prepare_data()
    configure_plot_style()
    plot_runtime(data, timeout_data)
    plot_memory(data, timeout_data)