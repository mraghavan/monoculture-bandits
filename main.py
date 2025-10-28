from src.simulation import simulate_single_agent, simulate_independent_agents
from src.agent import Agent
from src.bandit import Bandit
import numpy as np
import matplotlib.pyplot as plt

def estimate_agent_failure_prob(agent: Agent, bandit: Bandit, num_samples: int = 1000) -> float:
    """
    Estimates the failure probability of a single agent by sampling its choices.
    """
    if not isinstance(agent, Agent) or not hasattr(agent, 'choose_arm'):
        raise TypeError("The 'agent' argument must be an instance of Agent with a 'choose_arm' method.")

    failures = 0
    for _ in range(num_samples):
        if agent.choose_arm() != bandit.best_arm:
            failures += 1
    return failures / num_samples

def main():
    num_trials = 1000
    num_steps = 1000
    num_agents = 5
    N0_values = [1, 5, 10]

    results = {}

    for N0 in N0_values:
        # Run single-agent simulations
        single_agent_sims = [simulate_single_agent(num_steps, N0) for _ in range(num_trials)]

        # Estimate failure probability for each trial's final agent state
        per_trial_p = np.array([estimate_agent_failure_prob(agent, bandit) for bandit, agent in single_agent_sims])

        # Run independent agent simulations
        independent_agents_outcomes = np.array([simulate_independent_agents(num_agents, num_steps, N0) for _ in range(num_trials)])

        # Calculate the statistics
        p_mean = np.mean(per_trial_p)
        p_std_err = np.std(per_trial_p) / np.sqrt(num_trials)

        per_trial_p_n = per_trial_p ** num_agents
        p_n_mean = np.mean(per_trial_p_n)
        p_n_std_err = np.std(per_trial_p_n) / np.sqrt(num_trials)

        results[N0] = {
            'Single Agent': {
                'failure_rate': p_mean,
                'std_err': p_std_err
            },
            'Independent Agents': {
                'failure_rate': 1 - np.mean(independent_agents_outcomes),
                'std_err': np.std(independent_agents_outcomes) / np.sqrt(num_trials)
            },
            'Theoretical p^n': {
                'failure_rate': p_n_mean,
                'std_err': p_n_std_err
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
