import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os
import glob
import numpy as np

def configure_plot_style():
    # CONSTANTS AND SETTINGS
    # ======================
    ALGORITHM_STYLES = {
        'IFI': {'color': '#3366cc', 'marker': 'o', 'linestyle': '-', 'markersize': 9},
        'FIT': {'color': '#dc3912', 'marker': 's', 'linestyle': '--', 'markersize': 8},
        'BF': {'color': '#ff9900', 'marker': '^', 'linestyle': '-.', 'markersize': 9},
        'IESJ': {'color': '#109618', 'marker': 'D', 'linestyle': ':', 'markersize': 8},
        'RNLJ': {'color': '#990099', 'marker': 'v', 'linestyle': '-', 'markersize': 9}
    }

    PLOT_SETTINGS = {
        'figsize': (18, 6),
        'title_fontsize': 18,
        'label_fontsize': 18,
        'legend_fontsize': 18,
        'grid_alpha': 0.2,
        'dpi': 300,
        'legend_cols': 5
    }

    return ALGORITHM_STYLES, PLOT_SETTINGS

# ======================
# DATA LOADING
# ======================
def load_data():
    csv_files = glob.glob('Exp_results/RealWorld/*.csv')
    dfs = []
    
    for file in csv_files:
        parts = os.path.splitext(file)[0].split('/')
        parts = parts[-1].split('_')
        alg = parts[0]
        dataset = parts[-1].split('.')[0]
        
        df = pd.read_csv(file, header=None)
        df.columns = ['algorithm', 'size', 'epsilon', 'runtime', 'memory', 'K']
        df['algorithm'] = alg
        df['dataset'] = dataset
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)



# ======================
# PLOTTING FUNCTION
# ======================
def create_dataset_plots(data, y_col, y_label, log_scale=False, show_legend=False):

    
    datasets = sorted(data['dataset'].unique())
    ALGORITHM_STYLES ,PLOT_SETTINGS = configure_plot_style()
    
    fig, axs = plt.subplots(
        1, len(datasets),
        figsize=PLOT_SETTINGS['figsize'],
        dpi=PLOT_SETTINGS['dpi'],
        sharey=True
    )
    
    if len(datasets) == 1:
        axs = [axs]
    
    for j, dataset in enumerate(datasets):
        ax = axs[j]
        subset = data[data['dataset'] == dataset]
        size = subset['size'].iloc[0]
        
        for alg, alg_data in subset.groupby('algorithm'):
            style = ALGORITHM_STYLES[alg]
            ax.plot(
                alg_data['epsilon'],
                alg_data[y_col],
                label=alg if show_legend else None,  # Only set label if showing legend
                color=style['color'],
                marker=style['marker'],
                linestyle=style['linestyle'],
                markersize=style['markersize'],
                linewidth=2
            )
        
        if log_scale:
            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(ScalarFormatter())
        
        ax.set_xlabel('ε', fontsize=PLOT_SETTINGS['label_fontsize'])
        ax.set_ylabel(y_label if j == 0 else '', fontsize=PLOT_SETTINGS['label_fontsize'])
        ax.set_title(f'{dataset} (n={size})', fontsize=PLOT_SETTINGS['title_fontsize'])
        ax.grid(alpha=PLOT_SETTINGS['grid_alpha'])
    
    if show_legend:
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.15),
            ncol=PLOT_SETTINGS['legend_cols'],
            fontsize=PLOT_SETTINGS['legend_fontsize']
        )
        #plt.subplots_adjust(bottom=0.25)
    else:
        plt.tight_layout()
    
    return fig

# ======================
# CREATE AND SAVE PLOTS
# ======================

if __name__ == "__main__":

    data = load_data()

    # ======================

    # Runtime plots (no legend)
    runtime_fig = create_dataset_plots(
        data,
        y_col='runtime',
        y_label='Runtime (s)',
        log_scale=True,
        show_legend=False  # No legend for runtime
    )
    runtime_fig.savefig('runtime_by_dataset.png', bbox_inches='tight')

    # Memory plots (with legend)
    memory_fig = create_dataset_plots(
        data,
        y_col='memory',
        y_label='Memory (MB)',
        log_scale=False,
        show_legend=True  # Show legend for memory
    )
    memory_fig.savefig('memory_by_dataset.png', bbox_inches='tight')

    plt.close('all')