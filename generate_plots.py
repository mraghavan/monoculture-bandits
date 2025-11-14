import json
import os
import matplotlib.pyplot as plt
import numpy as np

def generate_plots():
    results_dir = 'results'
    plots_dir = 'plots'

    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(results_dir, filename)
            with open(filepath, 'r') as f:
                results = json.load(f)

            # Extract N0 values and conditions
            N0_values = sorted([int(n) for n in results.keys()])
            conditions = sorted(results[str(N0_values[0])].keys())

            # Plotting
            fig, ax = plt.subplots(figsize=(12, 8))

            for condition in conditions:
                failure_rates = [results[str(N0)][condition]['failure_rate'] for N0 in N0_values]
                std_errs = [results[str(N0)][condition]['std_err'] for N0 in N0_values]
                ax.errorbar(N0_values, failure_rates, yerr=std_errs, marker='o', linestyle='-', label=condition, capsize=5)

            ax.set_xlabel('N0 (Initial Samples per Arm)')
            ax.set_ylabel('Failure Rate')
            ax.set_title(f'Failure Rate vs. N0 ({filename})')
            ax.set_xticks(N0_values)
            ax.legend()
            ax.grid(True)

            plot_filename = os.path.splitext(filename)[0] + '.png'
            plt.savefig(os.path.join(plots_dir, plot_filename))
            plt.close(fig)
            print(f"Generated plot: {plot_filename}")

if __name__ == "__main__":
    generate_plots()
