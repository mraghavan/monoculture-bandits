import numpy as np
import argparse

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
        total_regret = 0
        for _ in range(self.n_rounds):
            available_arms = list(range(self.n_arms))
            pulled_arms = {}
            rewards = {}

            for agent_idx, agent in enumerate(self.agents):
                chosen_arm = agent.choose_arm(available_arms)
                if chosen_arm != -1:
                    pulled_arms[agent_idx] = chosen_arm
                    available_arms.remove(chosen_arm)
                    reward = self.arms[chosen_arm].pull()
                    rewards[agent_idx] = reward

            true_means = [arm.mean for arm in self.arms]
            sorted_true_means = sorted(true_means, reverse=True)
            optimal_reward = sum(sorted_true_means[:self.n_agents])
            actual_reward = sum(rewards.values())
            regret = optimal_reward - actual_reward
            total_regret += regret

            for agent in self.agents:
                for agent_idx, arm_idx in pulled_arms.items():
                    agent.update(arm_idx, rewards[agent_idx])
        return total_regret

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run bandit simulations.")
    parser.add_argument("-n", "--n_agents", type=int, default=9, help="Number of agents")
    parser.add_argument("-k", "--n_arms", type=int, default=10, help="Number of arms")
    parser.add_argument("-t", "--n_rounds", type=int, default=100, help="Number of rounds")
    parser.add_argument("--num_simulations", type=int, default=1000, help="Number of simulations to average over")
    args = parser.parse_args()

    total_monoculture_regret = 0
    total_polyculture_regret = 0

    print(f"Running {args.num_simulations} simulations...")

    for i in range(args.num_simulations):
        np.random.seed(i)
        arms = [Arm(np.random.normal(0, 1), 1) for _ in range(args.n_arms)]

        monoculture_sim = Simulation(args.n_agents, args.n_arms, args.n_rounds, is_monoculture=True, arms=arms)
        total_monoculture_regret += monoculture_sim.run()

        polyculture_sim = Simulation(args.n_agents, args.n_arms, args.n_rounds, is_monoculture=False, arms=arms)
        total_polyculture_regret += polyculture_sim.run()

    avg_monoculture_regret = total_monoculture_regret / args.num_simulations
    avg_polyculture_regret = total_polyculture_regret / args.num_simulations

    print(f"\nAverage Total Regret (over {args.num_simulations} runs):")
    print(f"  Monoculture: {avg_monoculture_regret}")
    print(f"  Polyculture: {avg_polyculture_regret}")
