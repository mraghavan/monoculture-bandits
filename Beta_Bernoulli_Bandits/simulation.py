import numpy as np
from scipy.stats import beta

class Arm:
    def __init__(self, alpha=2.0, beta_param=2.0):
        self.alpha = alpha
        self.beta = beta_param
        self.p = np.random.beta(self.alpha, self.beta)

    def pull(self):
        return np.random.binomial(1, self.p)

class Agent:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.beliefs = {i: (2.0, 2.0) for i in range(n_arms)}

    def expected_reward(self, arm_index):
        alpha, beta_param = self.beliefs[arm_index]
        return alpha / (alpha + beta_param)

    def update_belief(self, arm_index, reward):
        alpha, beta_param = self.beliefs[arm_index]
        if reward == 1:
            self.beliefs[arm_index] = (alpha + 1, beta_param)
        else:
            self.beliefs[arm_index] = (alpha, beta_param + 1)

class Simulation:
    def __init__(self, n_agents, n_arms, n_rounds, setting, arms, n_0=1):
        self.n_agents = n_agents
        self.n_arms = n_arms
        self.n_rounds = n_rounds
        self.setting = setting
        self.n_0 = n_0
        self.arms = arms
        self.agents = self._create_agents()
        self.pulled_arms_history = []

    def _create_agents(self):
        agents = [Agent(self.n_arms) for _ in range(self.n_agents)]

        if self.setting == 'monoculture':
            initial_beliefs = {}
            for i in range(self.n_arms):
                alpha, beta_param = 2.0, 2.0
                for _ in range(self.n_0):
                    reward = self.arms[i].pull()
                    if reward == 1:
                        alpha += 1
                    else:
                        beta_param += 1
                initial_beliefs[i] = (alpha, beta_param)

            for agent in agents:
                agent.beliefs = initial_beliefs.copy()
        else:  # Polyculture settings
            for agent in agents:
                for i in range(self.n_arms):
                    alpha, beta_param = 2.0, 2.0
                    for _ in range(self.n_0):
                        reward = self.arms[i].pull()
                        if reward == 1:
                            alpha += 1
                        else:
                            beta_param += 1
                    agent.beliefs[i] = (alpha, beta_param)
        return agents

    def run(self):
        agent_order = list(range(self.n_agents))
        if self.setting == 'monoculture' or self.setting == 'polyculture-fixed':
            np.random.shuffle(agent_order)

        for t in range(self.n_rounds):
            if self.setting == 'polyculture-random':
                np.random.shuffle(agent_order)

            pulled_this_round = []
            rewards_this_round = {}
            available_arms = list(range(self.n_arms))

            for agent_idx in agent_order:
                agent = self.agents[agent_idx]

                best_arm = -1
                max_expected_reward = -1

                shuffled_available_arms = np.random.permutation(available_arms)

                for arm_idx in shuffled_available_arms:
                    expected_reward = agent.expected_reward(arm_idx)
                    if expected_reward > max_expected_reward:
                        max_expected_reward = expected_reward
                        best_arm = arm_idx

                if best_arm != -1:
                    reward = self.arms[best_arm].pull()
                    pulled_this_round.append(best_arm)
                    rewards_this_round[best_arm] = reward
                    available_arms.remove(best_arm)

            self.pulled_arms_history.append(pulled_this_round)

            # Update beliefs
            for agent in self.agents:
                for arm_idx, reward in rewards_this_round.items():
                    agent.update_belief(arm_idx, reward)

    def calculate_bayesian_regret(self):
        total_regret = 0

        true_rewards = [arm.p for arm in self.arms]
        best_arm_indices = np.argsort(true_rewards)[-self.n_agents:]
        optimal_reward = np.sum([true_rewards[i] for i in best_arm_indices])

        for t in range(self.n_rounds):

            pulled_rewards = np.sum([true_rewards[i] for i in self.pulled_arms_history[t]])
            total_regret += (optimal_reward - pulled_rewards)

        return total_regret
