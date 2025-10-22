import unittest
import numpy as np
from src.bandit import Bandit

class TestBandit(unittest.TestCase):
    def test_bandit_initialization(self):
        bandit = Bandit(p_arms=[0.2, 0.8])
        self.assertEqual(len(bandit.p_arms), 2)
        self.assertEqual(bandit.best_arm, 1)

    def test_bandit_pull(self):
        bandit = Bandit(p_arms=[0.0, 1.0])
        self.assertEqual(bandit.pull(0), 0)
        self.assertEqual(bandit.pull(1), 1)

if __name__ == '__main__':
    unittest.main()
