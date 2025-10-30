from src.simulation import simulate_monoculture, simulate_polyculture
import numpy as np
import matplotlib.pyplot as plt

def main():
    num_trials = 1000
    num_steps = 1000
    num_agents = 2
    N0_values = [1, 5, 10, 20, 50, 100]

    results = {}

    for N0 in N0_values:
        monoculture_outcomes = np.array([simulate_monoculture(num_steps, N0) for _ in range(num_trials)])
        polyculture_outcomes = np.array([simulate_polyculture(num_agents, num_steps, N0) for _ in range(num_trials)])

        results[N0] = {
            'Monoculture': {
                'failure_rate': 1 - np.mean(monoculture_outcomes),
                'std_err': np.std(monoculture_outcomes) / np.sqrt(num_trials)
            },
            'Polyculture': {
                'failure_rate': 1 - np.mean(polyculture_outcomes),
                'std_err': np.std(polyculture_outcomes) / np.sqrt(num_trials)
            }
        }

    for N0, data in results.items():
        print(f"Results for N0 = {N0}:")
        for condition, values in data.items():
            print(f"  {condition}: Failure Rate = {values['failure_rate']:.4f} +/- {values['std_err']:.4f}")

    # --- New Plotting Logic ---

    # Extract data for plotting
    monoculture_failure_rates = [results[N0]['Monoculture']['failure_rate'] for N0 in N0_values]
    monoculture_std_errs = [results[N0]['Monoculture']['std_err'] for N0 in N0_values]
    polyculture_failure_rates = [results[N0]['Polyculture']['failure_rate'] for N0 in N0_values]
    polyculture_std_errs = [results[N0]['Polyculture']['std_err'] for N0 in N0_values]

    # Create a figure and a set of subplots
    fig, ax = plt.subplots()

    # Plot Monoculture results
    ax.errorbar(N0_values, monoculture_failure_rates, yerr=monoculture_std_errs, marker='o', linestyle='-', label='Monoculture', capsize=5)

    # Plot Polyculture results
    ax.errorbar(N0_values, polyculture_failure_rates, yerr=polyculture_std_errs, marker='o', linestyle='-', label='Polyculture', capsize=5)

    ax.set_xlabel('N0 (Initial Samples)')
    ax.set_ylabel('Failure Rate')
    ax.set_title('Failure Rate vs. Initial Samples (N0)')
    ax.legend()
    ax.grid(True)
    plt.savefig('failure_rates.png')
    plt.show()

if __name__ == "__main__":
    main()
