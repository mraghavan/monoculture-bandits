import unittest
import numpy as np
from src.agent import Agent
from src.bandit import Bandit

class TestAgent(unittest.TestCase):
    def test_agent_initialization(self):
        bandit = Bandit(num_arms=2, p_arms=[0.25, 0.75])
        agent = Agent(num_arms=2, N0=5, bandit=bandit)
        self.assertEqual(agent.num_arms, 2)
        # For a frequentist agent, the total number of pulls recorded in the beliefs
        # array should be equal to the number of arms times N0.
        expected_sum = 2 * 5
        self.assertEqual(np.sum(agent.beliefs), expected_sum)

    def test_choose_arm(self):
        agent = Agent(num_arms=3, N0=0)
        # Set beliefs to create a tie for the best arm
        agent.beliefs = np.array([[10, 1], [5, 5], [10, 1]])

        # Run choose_arm multiple times to check for random tie-breaking
        choices = [agent.choose_arm() for _ in range(100)]

        # The choices should be either arm 0 or arm 2
        self.assertTrue(all(c in [0, 2] for c in choices))
        # Both arms should be chosen at least once, indicating randomness
        self.assertTrue(0 in choices and 2 in choices)

    def test_update_belief(self):
        agent = Agent(num_arms=2, N0=0)
        agent.update_belief(0, 1)  # Success for arm 0
        self.assertEqual(agent.beliefs[0, 0], 1)
        self.assertEqual(agent.beliefs[0, 1], 0)

        agent.update_belief(0, 0)  # Failure for arm 0
        self.assertEqual(agent.beliefs[0, 0], 1)
        self.assertEqual(agent.beliefs[0, 1], 1)

if __name__ == '__main__':
    unittest.main()
