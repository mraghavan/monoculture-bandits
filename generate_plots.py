import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict

def load_and_group_results(results_dir):
    """
    Loads all JSON results from the given directory and groups them by
    simulation parameters (num_arms, num_steps).
    """
    grouped_results = defaultdict(list)
    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(results_dir, filename)
            with open(filepath, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from {filename}. Skipping.")
                    continue

                if 'params' not in data:
                    print(f"Warning: Missing 'params' key in {filename}. Skipping.")
                    continue

                params = data['params']
                if 'num_arms' not in params or 'num_steps' not in params:
                    print(f"Warning: Missing 'num_arms' or 'num_steps' in params for {filename}. Skipping.")
                    continue

                key = (params['num_arms'], params['num_steps'])
                grouped_results[key].append(data)
    return grouped_results

def generate_plots():
    """
    Generates plots from the aggregated simulation results.
    """
    results_dir = 'results'
    plots_dir = 'plots'

    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    grouped_results = load_and_group_results(results_dir)

    for (num_arms, num_steps), results_list in grouped_results.items():
        # Use pandas for easier data manipulation
        df = pd.json_normalize(results_list)

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 8))

        # Get unique conditions (monoculture, polyculture_2, etc.)
        conditions = df['params.condition'].unique()

        for condition in sorted(conditions):
            if condition == 'polyculture':
                # Group polyculture results by number of agents
                poly_groups = df[df['params.condition'] == 'polyculture'].groupby('params.num_agents')
                for num_agents, group in poly_groups:
                    group = group.sort_values('params.N0')
                    N0_values = group['params.N0']
                    failure_rates = group['results.failure_rate']
                    std_errs = group['results.std_err']
                    ax.errorbar(N0_values, failure_rates, yerr=std_errs, marker='o', linestyle='-', label=f'polyculture (k={num_agents})', capsize=5)
            else:
                # Plot monoculture results
                mono_group = df[df['params.condition'] == 'monoculture'].sort_values('params.N0')
                N0_values = mono_group['params.N0']
                failure_rates = mono_group['results.failure_rate']
                std_errs = mono_group['results.std_err']
                ax.errorbar(N0_values, failure_rates, yerr=std_errs, marker='o', linestyle='-', label='monoculture', capsize=5)

        ax.set_xlabel('N0 (Initial Samples per Arm)')
        ax.set_ylabel('Failure Rate')
        ax.set_title(f'Failure Rate vs. N0 (arms={num_arms}, steps={num_steps})')
        ax.legend()
        ax.grid(True)

        plot_filename = f"arms_{num_arms}_steps_{num_steps}_failure_rate.png"
        plt.savefig(os.path.join(plots_dir, plot_filename))
        plt.close(fig)
        print(f"Generated plot: {plot_filename}")

if __name__ == "__main__":
    generate_plots()
