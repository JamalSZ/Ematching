import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re
import numpy as np

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
    'label_fontsize': 12,
    'legend_fontsize': 10,
    'grid_alpha': 0.2,
    'dpi': 300  # Higher quality for saved images
}
# ==================================================================

def load_data():
    """Load all CSV files into a dictionary with e values as keys"""
    file_pattern = "Results/BP/exp_bp_*.csv"
    csv_files = glob.glob(file_pattern)
    
    data_dict = {}
    for file in csv_files:
        match = re.search(r'exp_bp_([0-9.]+)\.csv', file)
        if match:
            e = float(match.group(1))
            data_dict[e] = pd.read_csv(file)
    #print(data_dict)
    return data_dict

def create_plots(data_dict):
    """Generate consistently styled plots for all e values"""
    os.makedirs("Results/plots", exist_ok=True)
    
    for e, df in sorted(data_dict.items()):
        plt.figure(figsize=PLOT_SETTINGS['figsize'], dpi=PLOT_SETTINGS['dpi'])
        
        # Plot each algorithm with its designated style
        for algo, style in ALGORITHM_STYLES.items():
            if algo in df['algorithm'].unique():
                algo_data = df[df['algorithm'] == algo].sort_values('time_limit')
                plt.plot(algo_data['time_limit'], algo_data['max_size'], 
                         label=algo, 
                         color=style['color'],
                         marker=style['marker'],
                         linestyle=style['linestyle'],
                         markersize=style['markersize'],
                         linewidth=2.5)
        
        # Formatting
        plt.title(f"Performance Comparison (e = {e})", fontsize=PLOT_SETTINGS['title_fontsize'], pad=20)
        plt.xlabel("Time Limit", fontsize=PLOT_SETTINGS['label_fontsize'])
        plt.ylabel("Maximum Size", fontsize=PLOT_SETTINGS['label_fontsize'])
        
        # Create legend with consistent order
        handles, labels = plt.gca().get_legend_handles_labels()

        # Filter to only include known algorithms
        filtered = [(h, l) for h, l in zip(handles, labels) if l in ALGORITHM_STYLES]

        if filtered:
            # Sort according to ALGORITHM_STYLES order
            sorted_pairs = sorted(filtered, key=lambda x: list(ALGORITHM_STYLES.keys()).index(x[1]))
            ordered_handles, ordered_labels = zip(*sorted_pairs)
            plt.legend(ordered_handles, ordered_labels,
                       fontsize=PLOT_SETTINGS['legend_fontsize'],
                       bbox_to_anchor=(1.05, 1),
                       loc='upper left')
        else:
            # Fallback if no known algorithms are found
            plt.legend(fontsize=PLOT_SETTINGS['legend_fontsize'],
                       bbox_to_anchor=(1.05, 1),
                       loc='upper left')

        plt.grid(True, alpha=PLOT_SETTINGS['grid_alpha'])
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"Results/plots/tnperformance_e_{e:.2f}.png".replace('.', '_')  # handles decimals in filename
        plt.savefig(plot_filename, bbox_inches='tight', dpi=PLOT_SETTINGS['dpi'])
        plt.close()
        print(f"Created: {plot_filename}")
# ==================================================================

# Epsilons we want to plot
SELECTED_EPSILONS = [0.01, 0.02,0.03]

def load_data():
    """Load all CSV files into a dictionary with epsilon values as keys"""
    file_pattern = "Results/BP/exp_bp_*.csv"
    csv_files = glob.glob(file_pattern)
    
    data_dict = {}
    for file in csv_files:
        match = re.search(r'exp_bp_([0-9.]+)\.csv', file)
        if match:
            e = float(match.group(1))
            data_dict[e] = pd.read_csv(file)
    return data_dict

def create_combined_plot(data_dict):
    """Create a single figure with 4 subplots for selected epsilon values"""
    os.makedirs("Results/plots", exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=PLOT_SETTINGS['figsize'], dpi=PLOT_SETTINGS['dpi'], sharey=True)

    # Keep track of handles and labels for common legend
    legend_handles_labels = []

    for idx, e in enumerate(SELECTED_EPSILONS):
        ax = axes[idx]
        df = data_dict.get(e)
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
    output_filename = "Results/plots/combined_performance_selected_epsilons.png"
    plt.savefig(output_filename, bbox_inches='tight', dpi=PLOT_SETTINGS['dpi'])
    plt.close()
    print(f"Combined plot saved to {output_filename}")

if __name__ == "__main__":
    print("Starting plot generation...")
    data = load_data()
    print(f"Loaded data for {len(data)} different ε values")
    create_combined_plot(data)
    print("\nAll plots generated successfully!")

    create_plots(data)
    print("\nAll plots generated successfully!")