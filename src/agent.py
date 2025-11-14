import numpy as np

class Agent:
    def __init__(self, num_arms=2, N0=1, bandit=None):
        self.num_arms = num_arms
        # Beta distribution parameters (alpha, beta). A Beta(1,1) distribution
        # is a uniform distribution on [0,1].
        self.beliefs = np.ones((num_arms, 2))
        self.N0 = N0
        if bandit:
            self._initialize_prior_from_bandit(bandit)

    def _initialize_prior_from_bandit(self, bandit):
        """
        Initializes the agent's prior beliefs by sampling from a given bandit.
        For each of the N0 initial pulls for each arm, the agent observes a
        reward from the bandit and updates its beliefs accordingly.
        """
        for arm in range(self.num_arms):
            for _ in range(self.N0):
                reward = bandit.pull(arm)
                self.update_belief(arm, reward)

    def choose_arm(self):
        # The agent is greedy, always choosing the arm with the highest
        # expected reward based on its current beliefs. The expected reward
        # for a Beta distribution is alpha / (alpha + beta).
        expected_rewards = self.beliefs[:, 0] / (self.beliefs[:, 0] + self.beliefs[:, 1])
        return np.argmax(expected_rewards)

    def update_belief(self, arm_index, reward):
        # A reward of 1 corresponds to a success, so we increment alpha.
        # A reward of 0 corresponds to a failure, so we increment beta.
        if reward == 1:
            self.beliefs[arm_index, 0] += 1
        else:
            self.beliefs[arm_index, 1] += 1
