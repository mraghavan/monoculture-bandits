import numpy as np
import argparse
from collections import defaultdict
from scipy.stats import beta

def gini(x):
    """Calculate the Gini coefficient of a numpy array."""
    x = np.asarray(x)
    if np.amin(x) < 0:
        raise ValueError("Input array must be non-negative")
    x = np.sort(x)
    n = len(x)
    if n == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * x)) / (n * np.sum(x)) if np.sum(x) != 0 else 0.0

class Arm:
    def __init__(self, p):
        self.p = p # Probability of success

    def pull(self):
        return np.random.binomial(1, self.p)

class Agent:
    def __init__(self, n_arms, initial_priors=None):
        self.n_arms = n_arms
        if initial_priors:
            self.priors = [p.copy() for p in initial_priors]
        else:
            # Default is an uninformative prior Beta(1, 1)
            self.priors = [{'alpha': 1, 'beta': 1} for _ in range(n_arms)]

    def choose_arm(self, available_arms):
        best_arm = -1
        max_expected_reward = -1
        if not available_arms:
            return best_arm

        for arm_idx in available_arms:
            # Expected value of a Beta distribution is alpha / (alpha + beta)
            expected_value = self.priors[arm_idx]['alpha'] / (self.priors[arm_idx]['alpha'] + self.priors[arm_idx]['beta'])
            if expected_value > max_expected_reward:
                max_expected_reward = expected_value
                best_arm = arm_idx
        return best_arm

    def update(self, arm_idx, result):
        # Beta-Bernoulli update
        if result == 1: # Success
            self.priors[arm_idx]['alpha'] += 1
        else: # Failure
            self.priors[arm_idx]['beta'] += 1

class Simulation:
    def __init__(self, n_agents, n_arms, n_rounds, is_monoculture=True, arms=None):
        self.n_agents = n_agents
        self.n_arms = n_arms
        self.n_rounds = n_rounds
        self.is_monoculture = is_monoculture

        if arms:
            self.arms = arms
        else:
            # Arms now have a random probability of success
            self.arms = [Arm(p=np.random.uniform(0, 1)) for _ in range(n_arms)]

        self.agents = []
        if is_monoculture:
            # All agents start with the same uninformative prior Beta(1, 1)
            priors = [{'alpha': 1, 'beta': 1} for _ in range(n_arms)]
            for _ in range(n_agents):
                self.agents.append(Agent(n_arms, initial_priors=priors))
        else:
            # Polyculture agents have diverse, randomized priors
            for _ in range(n_agents):
                # Priors are randomized but keep the same mean as Beta(1,1) to be fair
                alpha = np.random.uniform(0.5, 1.5)
                beta_val = alpha
                priors = [{'alpha': alpha, 'beta': beta_val} for _ in range(n_arms)]
                self.agents.append(Agent(n_arms, initial_priors=priors))

    def run(self):
        total_realized_regret = 0
        total_bayesian_regret = 0
        best_arm_pulled_count = 0
        worst_arm_pulled_count = 0
        arm_pull_counts = np.zeros(self.n_arms)

        true_means = np.array([arm.p for arm in self.arms]) # True mean is just the probability p
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

            for arm_idx in pulled_arms_indices:
                arm_pull_counts[arm_idx] += 1

            actual_reward = sum(rewards.values())
            total_realized_regret += optimal_reward - actual_reward

            expected_reward_of_pulled = sum(true_means[i] for i in pulled_arms_indices)
            total_bayesian_regret += optimal_reward - expected_reward_of_pulled

            if best_arm_index in pulled_arms_indices:
                best_arm_pulled_count += 1
            if worst_arm_index in pulled_arms_indices:
                worst_arm_pulled_count += 1

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
        arms = [Arm(p=np.random.uniform(0, 1)) for _ in range(args.n_arms)]

        monoculture_sim = Simulation(args.n_agents, args.n_arms, args.n_rounds, is_monoculture=True, arms=arms)
        mono_results = monoculture_sim.run()
        for key, value in mono_results.items():
            monoculture_totals[key] += value

        polyculture_sim = Simulation(args.n_agents, args.n_arms, args.n_rounds, is_monoculture=False, arms=arms)
        poly_results = polyculture_sim.run()
        for key, value in poly_results.items():
            polyculture_totals[key] += value

    monoculture_avg = {key: value / args.num_simulations for key, value in monoculture_totals.items()}
    polyculture_avg = {key: value / args.num_simulations for key, value in polyculture_totals.items()}

    print("\n--- Averaged Results ---")
    print("\nMonoculture:")
    for key, value in monoculture_avg.items():
        print(f"  {key.replace('_', ' ').title()}: {value:.4f}")

    print("\nPolyculture:")
    for key, value in polyculture_avg.items():
        print(f"  {key.replace('_', ' ').title()}: {value:.4f}")
