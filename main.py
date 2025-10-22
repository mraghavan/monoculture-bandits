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

        results[N0] = {
            'Single Agent': {
                'failure_rate': 1 - np.mean(single_agent_outcomes),
                'std_err': np.std(single_agent_outcomes) / np.sqrt(num_trials)
            },
            'Independent Agents': {
                'failure_rate': 1 - np.mean(independent_agents_outcomes),
                'std_err': np.std(independent_agents_outcomes) / np.sqrt(num_trials)
            }
        }

    for N0, data in results.items():
        print(f"Results for N0 = {N0}:")
        for condition, values in data.items():
            print(f"  {condition}: Failure Rate = {values['failure_rate']:.4f} +/- {values['std_err']:.4f}")

    # Plotting the results
    labels = list(results[N0_values[0]].keys())
    for N0 in N0_values:
        failure_rates = [results[N0][label]['failure_rate'] for label in labels]
        std_errs = [results[N0][label]['std_err'] for label in labels]
        plt.errorbar(labels, failure_rates, yerr=std_errs, marker='o', linestyle='None', label=f'N0 = {N0}', capsize=5)

    plt.xlabel('Condition')
    plt.ylabel('Failure Rate')
    plt.title('Failure Rate of Greedy Algorithm in a Bandit Problem')
    plt.legend()
    plt.grid(True)
    plt.savefig('failure_rates.png')
    plt.show()

if __name__ == "__main__":
    main()
