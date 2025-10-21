import unittest
import numpy as np
from src.agent import Agent

class TestAgent(unittest.TestCase):
    def test_agent_initialization(self):
        agent = Agent(num_arms=2, N0=5)
        self.assertEqual(agent.num_arms, 2)
        expected_sum = 2 * 2 + 2 * 5  # (alpha=1, beta=1) for each arm initially, plus N0 samples for each arm
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
