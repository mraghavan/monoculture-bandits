import numpy as np
import argparse
from collections import defaultdict

def gini(x):
    """Calculate the Gini coefficient of a numpy array."""
    # The array must be non-negative
    x = np.asarray(x)
    if np.amin(x) < 0:
        raise ValueError("Input array must be non-negative")
    # Values must be sorted
    x = np.sort(x)
    n = len(x)
    if n == 0:
        return 0.0 # Gini of an empty set is 0
    # Gini coefficient calculation
    index = np.arange(1, n + 1)
    # Formula for Gini coefficient
    return (np.sum((2 * index - n - 1) * x)) / (n * np.sum(x)) if np.sum(x) != 0 else 0.0


class Arm:
    def __init__(self, mean, std_dev):
        self.mean = mean
        self.std_dev = std_dev

    def pull(self):
        return np.random.normal(self.mean, self.std_dev)

class Agent:
    def __init__(self, n_arms, initial_priors=None):
        self.n_arms = n_arms
        if initial_priors:
            self.priors = [p.copy() for p in initial_priors]
        else:
            self.priors = [{'mean': 0, 'std_dev': 10, 'n_pulls': 0} for _ in range(n_arms)]

    def choose_arm(self, available_arms):
        best_arm = -1
        max_expected_reward = -np.inf

        if not available_arms:
            return best_arm

        for arm_idx in available_arms:
            if self.priors[arm_idx]['mean'] > max_expected_reward:
                max_expected_reward = self.priors[arm_idx]['mean']
                best_arm = arm_idx
        return best_arm

    def update(self, arm_idx, reward):
        prior_mean = self.priors[arm_idx]['mean']
        prior_std_dev = self.priors[arm_idx]['std_dev']
        prior_var = prior_std_dev**2
        reward_var = 1.0
        post_var = 1.0 / (1.0 / prior_var + 1.0 / reward_var)
        post_mean = (prior_mean / prior_var + reward / reward_var) * post_var
        self.priors[arm_idx]['mean'] = post_mean
        self.priors[arm_idx]['std_dev'] = np.sqrt(post_var)
        self.priors[arm_idx]['n_pulls'] += 1

class Simulation:
    def __init__(self, n_agents, n_arms, n_rounds, is_monoculture=True, arms=None):
        self.n_agents = n_agents
        self.n_arms = n_arms
        self.n_rounds = n_rounds
        self.is_monoculture = is_monoculture

        if arms:
            self.arms = arms
        else:
            self.arms = [Arm(np.random.normal(0, 1), 1) for _ in range(n_arms)]

        self.agents = []
        if is_monoculture:
            priors = [{'mean': 0, 'std_dev': 10, 'n_pulls': 0} for _ in range(n_arms)]
            for _ in range(n_agents):
                self.agents.append(Agent(n_arms, initial_priors=priors))
        else:
            for _ in range(n_agents):
                priors = [{'mean': np.random.normal(0, 1), 'std_dev': 10, 'n_pulls': 0} for _ in range(n_arms)]
                self.agents.append(Agent(n_arms, initial_priors=priors))

    def run(self):
        total_realized_regret = 0
        total_bayesian_regret = 0
        best_arm_pulled_count = 0
        worst_arm_pulled_count = 0
        arm_pull_counts = np.zeros(self.n_arms)

        true_means = np.array([arm.mean for arm in self.arms])
        best_arm_index = np.argmax(true_means)
        worst_arm_index = np.argmin(true_means)

        sorted_true_means = np.sort(true_means)[::-1]
        optimal_reward = np.sum(sorted_true_means[:self.n_agents])

        for _ in range(self.n_rounds):
            available_arms = list(range(self.n_arms))
            pulled_arms_indices = []
            rewards = {}

            for agent_idx, agent in enumerate(self.agents):
                chosen_arm = agent.choose_arm(available_arms)
                if chosen_arm != -1:
                    pulled_arms_indices.append(chosen_arm)
                    available_arms.remove(chosen_arm)
                    reward = self.arms[chosen_arm].pull()
                    rewards[chosen_arm] = reward

            # Update pull counts for Gini coefficient
            for arm_idx in pulled_arms_indices:
                arm_pull_counts[arm_idx] += 1

            # Update regrets
            actual_reward = sum(rewards.values())
            total_realized_regret += optimal_reward - actual_reward

            expected_reward_of_pulled = sum(true_means[i] for i in pulled_arms_indices)
            total_bayesian_regret += optimal_reward - expected_reward_of_pulled

            # Update best/worst arm pull counts
            if best_arm_index in pulled_arms_indices:
                best_arm_pulled_count += 1
            if worst_arm_index in pulled_arms_indices:
                worst_arm_pulled_count += 1

            # Update agents
            for agent in self.agents:
                for arm_idx, reward_val in rewards.items():
                    agent.update(arm_idx, reward_val)

        gini_coefficient = gini(arm_pull_counts)

        return {
            'total_realized_regret': total_realized_regret,
            'total_bayesian_regret': total_bayesian_regret,
            'frac_best_arm_pulled': best_arm_pulled_count / self.n_rounds,
            'frac_worst_arm_pulled': worst_arm_pulled_count / self.n_rounds,
            'gini_coefficient': gini_coefficient,
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run bandit simulations.")
    parser.add_argument("-n", "--n_agents", type=int, default=9, help="Number of agents")
    parser.add_argument("-k", "--n_arms", type=int, default=10, help="Number of arms")
    parser.add_argument("-t", "--n_rounds", type=int, default=100, help="Number of rounds")
    parser.add_argument("--num_simulations", type=int, default=10, help="Number of simulations to average over")
    args = parser.parse_args()

    monoculture_totals = defaultdict(float)
    polyculture_totals = defaultdict(float)

    print(f"Running {args.num_simulations} simulations with n={args.n_agents}, k={args.n_arms}, t={args.n_rounds}...")

    for i in range(args.num_simulations):
        np.random.seed(i)
        arms = [Arm(np.random.normal(0, 1), 1) for _ in range(args.n_arms)]

        monoculture_sim = Simulation(args.n_agents, args.n_arms, args.n_rounds, is_monoculture=True, arms=arms)
        mono_results = monoculture_sim.run()
        for key, value in mono_results.items():
            monoculture_totals[key] += value

        polyculture_sim = Simulation(args.n_agents, args.n_arms, args.n_rounds, is_monoculture=False, arms=arms)
        poly_results = polyculture_sim.run()
        for key, value in poly_results.items():
            polyculture_totals[key] += value

    # Calculate averages
    monoculture_avg = {key: value / args.num_simulations for key, value in monoculture_totals.items()}
    polyculture_avg = {key: value / args.num_simulations for key, value in polyculture_totals.items()}

    print("\n--- Averaged Results ---")
    print("\nMonoculture:")
    for key, value in monoculture_avg.items():
        print(f"  {key.replace('_', ' ').title()}: {value:.4f}")

    print("\nPolyculture:")
    for key, value in polyculture_avg.items():
        print(f"  {key.replace('_', ' ').title()}: {value:.4f}")
