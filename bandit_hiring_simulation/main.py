import numpy as np
import argparse
from collections import defaultdict
import random

def gini(x):
    x = np.asarray(x)
    if np.amin(x) < 0:
        raise ValueError("Input array must be non-negative")
    x = np.sort(x)
    n = len(x)
    if n == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * x)) / (n * np.sum(x)) if np.sum(x) != 0 else 0.0

# --- Gaussian Model ---
class GaussianArm:
    def __init__(self, mean, std_dev=1):
        self.mean = mean
        self.std_dev = std_dev
    def pull(self):
        return np.random.normal(self.mean, self.std_dev)

class GaussianAgent:
    def __init__(self, n_arms, initial_priors):
        self.n_arms = n_arms
        self.priors = [p.copy() for p in initial_priors]

    def choose_arm(self, available_arms):
        best_arm, max_expected_reward = -1, -np.inf
        for arm_idx in available_arms:
            if self.priors[arm_idx]['mean'] > max_expected_reward:
                max_expected_reward, best_arm = self.priors[arm_idx]['mean'], arm_idx
        return best_arm

    def update(self, arm_idx, reward):
        prior_mean, prior_std_dev = self.priors[arm_idx]['mean'], self.priors[arm_idx]['std_dev']
        prior_var, reward_var = prior_std_dev**2, 1.0
        post_var = 1.0 / (1.0 / prior_var + 1.0 / reward_var)
        post_mean = (prior_mean / prior_var + reward / reward_var) * post_var
        self.priors[arm_idx].update({'mean': post_mean, 'std_dev': np.sqrt(post_var), 'n_pulls': self.priors[arm_idx]['n_pulls'] + 1})

# --- Bernoulli Model ---
class BernoulliArm:
    def __init__(self, p):
        self.p = p
    def pull(self):
        return np.random.binomial(1, self.p)

class BernoulliAgent:
    def __init__(self, n_arms, initial_priors):
        self.n_arms = n_arms
        self.priors = [p.copy() for p in initial_priors]

    def choose_arm(self, available_arms):
        best_arm, max_expected_reward = -1, -1
        for arm_idx in available_arms:
            p = self.priors[arm_idx]
            expected_value = p['alpha'] / (p['alpha'] + p['beta'])
            if expected_value > max_expected_reward:
                max_expected_reward, best_arm = expected_value, arm_idx
        return best_arm

    def update(self, arm_idx, result):
        if result == 1: self.priors[arm_idx]['alpha'] += 1
        else: self.priors[arm_idx]['beta'] += 1

# --- Simulation ---
class Simulation:
    def __init__(self, n_agents, n_arms, n_rounds, arms, model='gaussian', culture='monoculture'):
        self.n_agents, self.n_arms, self.n_rounds = n_agents, n_arms, n_rounds
        self.arms, self.model, self.culture = arms, model, culture

        AgentClass = GaussianAgent if model == 'gaussian' else BernoulliAgent
        self.agents = []

        if culture == 'monoculture':
            if model == 'gaussian':
                shared_priors = [{'mean': np.random.normal(0, 1), 'std_dev': 0.5, 'n_pulls': 0} for _ in range(n_arms)]
            else:
                shared_priors = []
                for _ in range(n_arms):
                    alpha = np.random.uniform(9.5, 10.5)
                    beta_val = 20 - alpha
                    shared_priors.append({'alpha': alpha, 'beta': beta_val})

            for _ in range(n_agents):
                self.agents.append(AgentClass(n_arms, initial_priors=shared_priors))

        else:
            for _ in range(n_agents):
                if model == 'gaussian':
                    priors = [{'mean': np.random.normal(0, 1), 'std_dev': 0.5, 'n_pulls': 0} for _ in range(n_arms)]
                else:
                    priors = []
                    for _ in range(n_arms):
                        alpha = np.random.uniform(9.5, 10.5)
                        beta_val = 20 - alpha
                        priors.append({'alpha': alpha, 'beta': beta_val})
                self.agents.append(AgentClass(n_arms, initial_priors=priors))

    def run(self):
        arm_pull_counts = np.zeros(self.n_arms)
        stats = defaultdict(float)

        true_means = np.array([arm.mean if self.model == 'gaussian' else arm.p for arm in self.arms])
        best_arm_index, worst_arm_index = np.argmax(true_means), np.argmin(true_means)
        optimal_reward = np.sum(np.sort(true_means)[::-1][:self.n_agents])

        agent_order = list(range(self.n_agents))
        for _ in range(self.n_rounds):
            if self.culture == 'polyculture_random':
                random.shuffle(agent_order)

            available_arms = list(range(self.n_arms))
            pulled_this_round, rewards = {}, {}

            for agent_idx in agent_order:
                agent = self.agents[agent_idx]
                chosen_arm = agent.choose_arm(available_arms)
                if chosen_arm != -1:
                    available_arms.remove(chosen_arm)
                    reward = self.arms[chosen_arm].pull()
                    pulled_this_round[agent_idx] = chosen_arm
                    rewards[chosen_arm] = reward

            pulled_indices = list(rewards.keys())
            arm_pull_counts[pulled_indices] += 1
            stats['total_realized_regret'] += optimal_reward - sum(rewards.values())
            stats['total_bayesian_regret'] += optimal_reward - sum(true_means[i] for i in pulled_indices)
            if best_arm_index in pulled_indices: stats['best_arm_pulled_count'] += 1
            if worst_arm_index in pulled_indices: stats['worst_arm_pulled_count'] += 1

            for agent in self.agents:
                for arm_idx, reward_val in rewards.items():
                    agent.update(arm_idx, reward_val)

        stats['frac_best_arm_pulled'] = stats['best_arm_pulled_count'] / self.n_rounds
        stats['frac_worst_arm_pulled'] = stats['worst_arm_pulled_count'] / self.n_rounds
        stats['gini_coefficient'] = gini(arm_pull_counts)
        return {k: v for k, v in stats.items() if 'count' not in k}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run bandit simulations.")
    parser.add_argument("-n", "--n_agents", type=int, default=90, help="Number of agents")
    parser.add_argument("-k", "--n_arms", type=int, default=100, help="Number of arms")
    parser.add_argument("-t", "--n_rounds", type=int, default=100, help="Number of rounds")
    parser.add_argument("--num_simulations", type=int, default=50, help="Number of simulations to average over")
    parser.add_argument("--model", type=str, default='gaussian', choices=['gaussian', 'bernoulli'], help="Statistical model for arms and agents")
    args = parser.parse_args()

    print(f"--- Running {args.model.upper()} Model (Randomized Monoculture Priors) ---")

    setups = ['monoculture', 'polyculture_fixed', 'polyculture_random']
    results = {setup: defaultdict(float) for setup in setups}

    for i in range(args.num_simulations):
        np.random.seed(i)
        random.seed(i)
        if args.model == 'gaussian':
            arms = [GaussianArm(np.random.normal(0, 1)) for _ in range(args.n_arms)]
        else:
            arms = [BernoulliArm(np.random.uniform(0, 1)) for _ in range(args.n_arms)]

        sims = {
            'monoculture': Simulation(args.n_agents, args.n_arms, args.n_rounds, arms, args.model, 'monoculture'),
            'polyculture_fixed': Simulation(args.n_agents, args.n_arms, args.n_rounds, arms, args.model, 'polyculture_fixed'),
            'polyculture_random': Simulation(args.n_agents, args.n_arms, args.n_rounds, arms, args.model, 'polyculture_random')
        }

        for setup_name, sim in sims.items():
            run_results = sim.run()
            for key, value in run_results.items():
                results[setup_name][key] += value

    print(f"\n--- Averaged Results over {args.num_simulations} runs ---")
    for setup_name, res in results.items():
        print(f"\n{setup_name.replace('_', ' ').title()}:")
        for key, value in res.items():
            print(f"  {key.replace('_', ' ').title()}: {value / args.num_simulations:.4f}")
