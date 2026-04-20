import numpy as np

class Bandit:
    def __init__(self, p_arms=None, num_arms=2, rng=None):
        self.rng = rng
        if p_arms is None:
            # The arm probabilities are drawn from a Beta(2,2) distribution.
            # This is a symmetric distribution on [0,1] that is less likely
            # to generate values very close to 0 or 1 than a uniform
            # distribution. This makes the learning problem more challenging.
            if self.rng is None:
                p_arms = np.random.beta(2, 2, size=num_arms)
            else:
                p_arms = self.rng.beta(2, 2, size=num_arms)
        self.p_arms = np.array(p_arms)
        self.best_arm = np.argmax(self.p_arms)

    def pull(self, arm_index):
        if self.rng is None:
            reward = np.random.rand()
        else:
            reward = self.rng.random()
        if reward < self.p_arms[arm_index]:
            return 1
        return 0
