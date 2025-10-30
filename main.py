from src.simulation import simulate_monoculture, simulate_polyculture
import numpy as np
import matplotlib.pyplot as plt

def main():
    num_trials = 1000
    num_steps = 1000
    num_agents = 5
    N0_values = [1, 5, 10]

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

    # Plotting the results
    labels = list(results[N0_values[0]].keys())
    width = 0.2  # the width of the bars

    # Create a figure and a set of subplots
    fig, ax = plt.subplots()

    # Iterate over N0_values to plot each as a separate group of bars
    for i, N0 in enumerate(N0_values):
        failure_rates = [results[N0][label]['failure_rate'] for label in labels]
        std_errs = [results[N0][label]['std_err'] for label in labels]

        # The x locations for the groups
        x = np.arange(len(labels)) + i * width

        ax.errorbar(x, failure_rates, yerr=std_errs, marker='o', linestyle='None', label=f'N0 = {N0}', capsize=5)

    ax.set_xlabel('Condition')
    ax.set_xticks(np.arange(len(labels)) + width * (len(N0_values) - 1) / 2)
    ax.set_xticklabels(labels)
    plt.ylabel('Failure Rate')
    plt.title('Failure Rate of Greedy Algorithm in a Bandit Problem')
    plt.legend()
    plt.grid(True)
    plt.savefig('failure_rates.png')
    plt.show()

if __name__ == "__main__":
    main()
