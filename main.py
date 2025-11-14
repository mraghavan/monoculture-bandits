from src.simulation import simulate_monoculture, simulate_polyculture
from src.bandit import Bandit
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
    num_trials = 10000
    num_steps = 1000
    num_agents = 3
    N0_values = [1, 5, 10, 20, 40, 60, 80, 100]

    results = {}

    for N0 in tqdm(N0_values, desc="Simulating N0 values"):
        # For each N0, create a fixed set of bandits for the trials
        bandits = [Bandit() for _ in range(num_trials)]

        monoculture_outcomes = np.array([simulate_monoculture(bandits[i], num_steps, N0) for i in range(num_trials)])
        polyculture_outcomes = np.array([simulate_polyculture(bandits[i], num_agents, num_steps, N0) for i in range(num_trials)])

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
    ax.errorbar(N0_values, polyculture_failure_rates, yerr=polyculture_std_errs, marker='o', linestyle='-', label=rf'Polyculture ($k={num_agents}$)', capsize=5)

    ax.set_xlabel('$N_0$ (Initial Samples)')
    ax.set_ylabel('Failure Rate')
    ax.set_title(f'$T = {num_steps}$')
    ax.legend()
    ax.grid(True)
    plt.savefig(f'failure_rates_T={num_steps}.png', dpi=600)
    plt.show()

if __name__ == "__main__":
    main()
