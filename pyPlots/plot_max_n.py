import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re
import numpy as np
EPSILONS = [0.01,0.51,1.51,2.51]
# ====================== CUSTOMIZATION SECTION ======================
# Define your 5 algorithms and their styles here
ALGORITHM_STYLES = {
    'IFI': {'color': '#3366cc', 'marker': 'o', 'linestyle': '-', 'markersize': 9},
    'FIT': {'color': '#dc3912', 'marker': 's', 'linestyle': '--', 'markersize': 8},
    'BF': {'color': '#ff9900', 'marker': '^', 'linestyle': '-.', 'markersize': 9},
    'IESJ': {'color': '#109618', 'marker': 'D', 'linestyle': ':', 'markersize': 8},
    'RNLJ': {'color': '#990099', 'marker': 'v', 'linestyle': '-', 'markersize': 9}
}

# Plot appearance settings
PLOT_SETTINGS = {
    'figsize': (11, 6),
    'title_fontsize': 14,
    'tick_fontsize': 16,
    'label_fontsize': 12,
    'legend_fontsize': 10,
    'grid_alpha': 0.2,
    'dpi': 300  # Higher quality for saved images
}
# ==================================================================

def load_data():
    """Load all CSV files into a dictionary with e values as keys"""
    file_pattern = "Exp_results/Exp_max_n/*.csv"
    csv_files = glob.glob(file_pattern)
    print("CSV files found:", csv_files)
    
    data_dict = {}
    for file in csv_files:
        match = re.search(r'([0-9.]+)\.csv', file)
        if match:
            e = float(match.group(1))
            data_dict[e] = pd.read_csv(file)
    #print(data_dict)
    return data_dict



def create_combined_plot(data_dict):
    """Create a single figure with 4 subplots for selected epsilon values"""
    os.makedirs("Results/plots", exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=PLOT_SETTINGS['figsize'], dpi=PLOT_SETTINGS['dpi'], sharey=True)

    # Keep track of handles and labels for common legend
    legend_handles_labels = []


    for idx, e in enumerate(EPSILONS):
        ax = axes[idx]
        df = data_dict.get(f"Exp_results/Exp_max_n/exp_bp_{e}.csv")
        if df is None:
            print(f"Warning: No data found for e = {e}")
            continue
        
        for algo, style in ALGORITHM_STYLES.items():
            if algo in df['algorithm'].unique():
                algo_data = df[df['algorithm'] == algo].sort_values('time_limit')
                line, = ax.plot(algo_data['time_limit'], algo_data['max_size']/10000,
                                label=algo,
                                color=style['color'],
                                marker=style['marker'],
                                linestyle=style['linestyle'],
                                markersize=style['markersize'],
                                linewidth=2.5)
                if idx == 0:  # Only collect handles once
                    legend_handles_labels.append((line, algo))

        ax.set_title(f"ε = {e}", fontsize=PLOT_SETTINGS['title_fontsize'])
        ax.set_xlabel("Time Limit", fontsize=PLOT_SETTINGS['label_fontsize'])
        if idx == 0:
            ax.set_ylabel("Maximum Size x 10000", fontsize=PLOT_SETTINGS['label_fontsize'])
        ax.grid(True, alpha=PLOT_SETTINGS['grid_alpha'])

        # Set tick label sizes
        ax.tick_params(axis='both', labelsize=PLOT_SETTINGS['tick_fontsize'])

    # Set a main title for the entire figure
    #fig.suptitle("Performance Comparison for Different ε Values", fontsize=PLOT_SETTINGS['title_fontsize'] + 2)

    # Create common legend below the subplots
    handles, labels = zip(*legend_handles_labels)
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=PLOT_SETTINGS['legend_fontsize'], frameon=False)

    # Adjust layout to leave space for legend
    plt.tight_layout(rect=[0, 0.15, 1, 0.9])  # [left, bottom, right, top]
    output_filename = "Plots/epsilons.png"
    plt.savefig(output_filename, bbox_inches='tight', dpi=PLOT_SETTINGS['dpi'])
    plt.close()
    print(f"Combined plot saved to {output_filename}")

if __name__ == "__main__":
    print("Starting plot generation...")
    data = load_data()
    print(f"Loaded data for {len(data)} different ε values")
    create_combined_plot(data)
    print("\nAll plots generated successfully!")
