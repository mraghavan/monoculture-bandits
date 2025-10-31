import numpy as np

class Bandit:
    def __init__(self, p_arms=None):
        if p_arms is None:
            p_arms = np.random.beta(2, 2, 2)
        self.p_arms = p_arms
        self.best_arm = np.argmax(p_arms)

    def pull(self, arm_index):
        if np.random.rand() < self.p_arms[arm_index]:
            return 1
        return 0
