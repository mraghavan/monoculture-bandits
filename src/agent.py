import numpy as np

class Agent:
    def __init__(self, num_arms=2, N0=1, bandit=None, rng=None):
        self.num_arms = num_arms
        self.rng = rng
        # Beliefs are now initialized to zero for a frequentist approach.
        # Column 0: successes, Column 1: failures
        self.beliefs = np.zeros((num_arms, 2))
        self.N0 = N0
        if bandit:
            self._initialize_beliefs_from_bandit(bandit)

    def _initialize_beliefs_from_bandit(self, bandit):
        """
        Initializes the agent's beliefs by sampling from a given bandit.
        For each of the N0 initial pulls for each arm, the agent observes a
        reward from the bandit and updates its beliefs accordingly.
        """
        for arm in range(self.num_arms):
            for _ in range(self.N0):
                reward = bandit.pull(arm)
                self.update_belief(arm, reward)

    def choose_arm(self):
        """
        Chooses an arm based on the highest expected reward.
        Handles cases where an arm has not been pulled and breaks ties randomly.
        """
        successes = self.beliefs[:, 0]
        failures = self.beliefs[:, 1]
        total_pulls = successes + failures

        # Avoid division by zero for arms that haven't been pulled
        expected_rewards = np.divide(successes, total_pulls, out=np.zeros_like(successes, dtype=float), where=total_pulls!=0)

        # Find the maximum expected reward
        max_reward = np.max(expected_rewards)

        # Get all arms with the maximum reward
        best_arms = np.where(expected_rewards == max_reward)[0]

        # Randomly choose one of the best arms to break ties
        if self.rng is None:
            return np.random.choice(best_arms)
        return self.rng.choice(best_arms)

    def update_belief(self, arm_index, reward):
        """
        Updates the agent's belief about an arm based on the received reward.
        """
        if reward == 1:
            self.beliefs[arm_index, 0] += 1  # Increment successes
        else:
            self.beliefs[arm_index, 1] += 1  # Increment failures
