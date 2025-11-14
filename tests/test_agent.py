import unittest
import numpy as np
from src.agent import Agent
from src.bandit import Bandit

class TestAgent(unittest.TestCase):
    def test_agent_initialization(self):
        bandit = Bandit(num_arms=2)
        agent = Agent(num_arms=2, N0=5, bandit=bandit)
        self.assertEqual(agent.num_arms, 2)
        # Each arm starts with beliefs (1,1). N0 pulls are added for each arm.
        # So for each arm, the sum of beliefs will be 1+1+5 = 7.
        # For 2 arms, the total sum is 2 * 7 = 14.
        expected_sum = 2 * (1 + 1 + 5)
        self.assertEqual(np.sum(agent.beliefs), expected_sum)

    def test_choose_arm(self):
        agent = Agent(N0=0)
        agent.beliefs = np.array([[1, 1], [10, 1]])
        self.assertEqual(agent.choose_arm(), 1)

    def test_update_belief(self):
        agent = Agent(N0=0)
        agent.update_belief(0, 1)
        self.assertEqual(agent.beliefs[0, 0], 2)
        agent.update_belief(0, 0)
        self.assertEqual(agent.beliefs[0, 1], 2)

if __name__ == '__main__':
    unittest.main()
