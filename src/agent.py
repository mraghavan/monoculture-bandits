import numpy as np

class Agent:
    def __init__(self, num_arms=2, N0=1, bandit=None):
        self.num_arms = num_arms
        self.beliefs = np.ones((num_arms, 2))  # Beta distribution parameters (alpha, beta)
        self.N0 = N0
        if bandit:
            self._initialize_prior(bandit)

    def _initialize_prior(self, bandit):
        for arm in range(self.num_arms):
            for _ in range(self.N0):
                reward = bandit.pull(arm)
                self.update_belief(arm, reward)

    def choose_arm(self):
        expected_rewards = self.beliefs[:, 0] / (self.beliefs[:, 0] + self.beliefs[:, 1])
        return np.argmax(expected_rewards)

    def update_belief(self, arm_index, reward):
        if reward == 1:
            self.beliefs[arm_index, 0] += 1
        else:
            self.beliefs[arm_index, 1] += 1
