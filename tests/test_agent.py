import unittest
import numpy as np
from src.agent import Agent

class TestAgent(unittest.TestCase):
    def test_agent_initialization(self):
        class MockBandit:
            def pull(self, arm_index):
                return 1

        agent = Agent(num_arms=2, N0=5, bandit=MockBandit())
        self.assertEqual(agent.num_arms, 2)
        # Each arm starts with beliefs (1, 1). N0=5 pulls are all rewarded (1), so we add 5 to alpha for each arm.
        expected_sum = (1 + 5 + 1) + (1 + 5 + 1)
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
