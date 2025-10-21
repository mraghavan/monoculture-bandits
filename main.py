from src.simulation import simulate_single_agent, simulate_shared_information, simulate_independent_agents
import numpy as np
import matplotlib.pyplot as plt

def main():
    num_trials = 1000
    num_steps = 1000
    num_agents = 5
    N0_values = [1, 5, 10]

    results = {}

    for N0 in N0_values:
        single_agent_correct = np.mean([simulate_single_agent(num_steps, N0) for _ in range(num_trials)])
        shared_info_correct = np.mean([simulate_shared_information(num_agents, num_steps, N0) for _ in range(num_trials)])
        independent_agents_correct = np.mean([simulate_independent_agents(num_agents, num_steps, N0) for _ in range(num_trials)])

        results[N0] = {
            'Single Agent': 1 - single_agent_correct,
            'Shared Information': 1 - shared_info_correct,
            'Independent Agents': 1 - independent_agents_correct
        }

    for N0, data in results.items():
        print(f"Results for N0 = {N0}:")
        for condition, failure_rate in data.items():
            print(f"  {condition}: Failure Rate = {failure_rate:.4f}")

    # Plotting the results
    labels = list(results[N0_values[0]].keys())
    for N0 in N0_values:
        failure_rates = [results[N0][label] for label in labels]
        plt.plot(labels, failure_rates, marker='o', label=f'N0 = {N0}')

    plt.xlabel('Condition')
    plt.ylabel('Failure Rate')
    plt.title('Failure Rate of Greedy Algorithm in a Bandit Problem')
    plt.legend()
    plt.grid(True)
    plt.savefig('failure_rates.png')
    plt.show()

if __name__ == "__main__":
    main()
