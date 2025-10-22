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
        # --- Single Agent ---
        single_agent_outcomes = [simulate_single_agent(num_steps, N0) for _ in range(num_trials)]
        single_agent_failures = 1 - np.array(single_agent_outcomes)

        # --- Independent Agents ---
        independent_agents_outcomes = [simulate_independent_agents(num_agents, num_steps, N0) for _ in range(num_trials)]
        independent_agents_failures = 1 - np.array(independent_agents_outcomes)

        results[N0] = {
            'Single Agent': {
                'failure_rate': np.mean(single_agent_failures),
                'sem': np.std(single_agent_failures) / np.sqrt(num_trials)
            },
            'Independent Agents': {
                'failure_rate': np.mean(independent_agents_failures),
                'sem': np.std(independent_agents_failures) / np.sqrt(num_trials)
            }
        }

    for N0, data in results.items():
        print(f"Results for N0 = {N0}:")
        for condition, values in data.items():
            print(f"  {condition}: Failure Rate = {values['failure_rate']:.4f} (SEM = {values['sem']:.4f})")

    # --- Plotting the results ---
    labels = list(results[N0_values[0]].keys())
    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars for jitter

    fig, ax = plt.subplots()

    # Define colors for different N0 values to make the plot clearer
    colors = {1: 'skyblue', 5: 'royalblue', 10: 'navy'}

    for i, N0 in enumerate(N0_values):
        failure_rates = [results[N0][label]['failure_rate'] for label in labels]
        sems = [results[N0][label]['sem'] for label in labels]
        offset = (i - len(N0_values) / 2) * width + width / 2

        # Use plt.errorbar with fmt='o' for points and no line
        ax.errorbar(x + offset, failure_rates, yerr=sems, fmt='o',
                    label=f'N0 = {N0}', capsize=5, color=colors.get(N0))

    ax.set_xlabel('Condition')
    ax.set_ylabel('Failure Rate')
    ax.set_title('Failure Rate of Greedy Algorithm in a Bandit Problem')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    fig.tight_layout()
    plt.savefig('failure_rates_with_uncertainty.png')
    plt.show()

if __name__ == "__main__":
    main()
