import json
import os
import matplotlib.pyplot as plt
import pandas as pd

def generate_plots():
    """
    Generates plots from the aggregated simulation results.
    """
    # Set larger font sizes for all plot elements
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 20
    })
    
    results_dir = 'results'
    plots_dir = 'plots'

    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(results_dir, filename)

            with open(filepath, 'r') as f:
                try:
                    results_list = json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from {filename}. Skipping.")
                    continue

            if not results_list:
                print(f"Warning: No data in {filename}. Skipping.")
                continue

            # Use pandas for easier data manipulation
            df = pd.json_normalize(results_list)

            # Get parameters from the first record for the title
            num_arms = df['params.num_arms'][0]
            num_steps = df['params.num_steps'][0]

            # Plotting
            fig, ax = plt.subplots(figsize=(12, 8))

            plot_series = []

            mono_group = df[df['params.condition'] == 'monoculture'].sort_values('params.N0')
            if not mono_group.empty:
                plot_series.append(('monoculture', mono_group))

            poly_groups = df[df['params.condition'] == 'polyculture'].groupby('params.num_agents')
            for num_agents, group in sorted(poly_groups, key=lambda item: int(item[0])):
                plot_series.append((f'polyculture ($k={int(num_agents)}$)', group.sort_values('params.N0')))

            line_styles_markers = [
                ('-', 'o'),
                ('--', 's'),
                ('-.', '^'),
                ((0, (4, 1.5, 1, 1.5)), 'D'),
                ((0, (5, 2)), 'v'),
                ((0, (3, 1, 1, 1)), 'P'),
            ]
            cmap = plt.get_cmap('cubehelix')
            num_series = len(plot_series)
            if num_series == 1:
                series_colors = [cmap(0.45)]
            else:
                min_color = 0.15
                max_color = 0.75
                series_colors = [
                    cmap(min_color + (max_color - min_color) * index / (num_series - 1))
                    for index in range(num_series)
                ]

            for index, (label, group) in enumerate(plot_series):
                linestyle, marker = line_styles_markers[index % len(line_styles_markers)]
                color = series_colors[index]
                N0_values = group['params.N0']
                failure_rates = group['results.failure_rate']
                std_errs = group['results.std_err']
                ax.errorbar(
                    N0_values,
                    failure_rates,
                    yerr=std_errs,
                    marker=marker,
                    linestyle=linestyle,
                    label=label,
                    color=color,
                    ecolor=color,
                    markerfacecolor=color,
                    markeredgecolor='black',
                    markeredgewidth=1.6,
                    markersize=7,
                    linewidth=2.3,
                    capsize=5,
                )

            ax.set_xlabel('$N_0$', fontsize=16)
            ax.set_ylabel('Failure Rate', fontsize=16)
            ax.set_title(f'Monoculture vs. polyculture failure', fontsize=18)
            ax.legend(fontsize=14)
            ax.tick_params(labelsize=14)
            ax.grid(True)

            plot_filename = os.path.splitext(filename)[0] + '.png'
            plt.savefig(os.path.join(plots_dir, plot_filename), dpi=600)
            plt.close(fig)
            print(f"Generated plot: {plot_filename}")

if __name__ == "__main__":
    generate_plots()
