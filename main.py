from src.simulation import simulate_single_agent, simulate_independent_agents
import numpy as np
import matplotlib.pyplot as plt

def main():
    num_trials = 1000
    num_steps = 1000
    num_agents = 5
    N0_values = [1, 5, 10]

    results = {}

    for N0 in N0_values:
        single_agent_outcomes = np.array([simulate_single_agent(num_steps, N0) for _ in range(num_trials)])
        independent_agents_outcomes = np.array([simulate_independent_agents(num_agents, num_steps, N0) for _ in range(num_trials)])

        p = 1 - np.mean(single_agent_outcomes)
        z = 1.96  # For 95% confidence interval
        p_ci = z * np.sqrt(p * (1 - p) / num_trials)

        results[N0] = {
            'Single Agent': {
                'failure_rate': p,
                'std_err': np.std(single_agent_outcomes) / np.sqrt(num_trials),
                'ci': p_ci
            },
            'Independent Agents': {
                'failure_rate': 1 - np.mean(independent_agents_outcomes),
                'std_err': np.std(independent_agents_outcomes) / np.sqrt(num_trials)
            },
            'Theoretical p^n': {
                'failure_rate': p**num_agents,
                'std_err': 0,  # No variance in a theoretical calculation
                'ci': (
                    p**num_agents - (p - p_ci)**num_agents,
                    (p + p_ci)**num_agents - p**num_agents
                )
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

        y_errs = []
        for label in labels:
            point_data = results[N0][label]
            if label == 'Theoretical p^n':
                # Use the asymmetric confidence interval
                y_errs.append(point_data['ci'])
            else:
                # Use the symmetric standard error
                y_errs.append((point_data['std_err'], point_data['std_err']))

        # Transpose to get shape (2, N) for matplotlib
        y_errs_transposed = np.array(y_errs).T

        # The x locations for the groups
        x = np.arange(len(labels)) + i * width

        ax.errorbar(x, failure_rates, yerr=y_errs_transposed, marker='o', linestyle='None', label=f'N0 = {N0}', capsize=5)

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
