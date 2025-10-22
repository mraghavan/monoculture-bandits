import numpy as np

class Agent:
    def __init__(self, num_arms=2, N0=1):
        self.num_arms = num_arms
        self.beliefs = np.ones((num_arms, 2))  # Beta distribution parameters (alpha, beta)
        self.N0 = N0
        self._initialize_prior()

    def _initialize_prior(self):
        for arm in range(self.num_arms):
            for _ in range(self.N0):
                if np.random.rand() < 0.5:
                    self.beliefs[arm, 0] += 1
                else:
                    self.beliefs[arm, 1] += 1

    def choose_arm(self):
        expected_rewards = self.beliefs[:, 0] / (self.beliefs[:, 0] + self.beliefs[:, 1])
        return np.argmax(expected_rewards)

    def update_belief(self, arm_index, reward):
        if reward == 1:
            self.beliefs[arm_index, 0] += 1
        else:
            self.beliefs[arm_index, 1] += 1
