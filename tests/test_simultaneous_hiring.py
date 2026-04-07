import unittest
import numpy as np
from Crowd_Wisdom_Simultaneous import deferred_acceptance

class TestSimultaneousHiring(unittest.TestCase):
    def test_deferred_acceptance_simple(self):
        # 2 candidates, 2 firms, capacity 1 each
        # Candidate 0 prefers Firm 0 > Firm 1
        # Candidate 1 prefers Firm 0 > Firm 1
        # Firm 0 prefers Candidate 1 > Candidate 0
        # Firm 1 prefers Candidate 1 > Candidate 0

        candidate_prefs = np.array([
            [0, 1], # C0
            [0, 1]  # C1
        ])

        # Firm preferences (ranks: lower is better)
        firm_prefs_rank = np.array([
            [1, 0], # F0: C1 is rank 0, C0 is rank 1
            [1, 0]  # F1: C1 is rank 0, C0 is rank 1
        ])

        firm_capacities = [1, 1]

        matched = deferred_acceptance(candidate_prefs, firm_prefs_rank, firm_capacities)

        # C1 proposes to F0. F0 accepts.
        # C0 proposes to F0. F0 compares C1 (rank 0) and C0 (rank 1). F0 keeps C1, rejects C0.
        # C0 proposes to F1. F1 accepts.
        # Result: F0 matched with C1, F1 matched with C0.

        self.assertIn(1, matched)
        self.assertIn(0, matched)
        self.assertEqual(len(matched), 2)

        # Check specific matches if possible (DA returns a flat list of matched candidates)
        # In my implementation, matched_candidates is just a list.
        # Let's check which firm they are matched to by modifying DA or just checking the set.
        self.assertSetEqual(set(matched), {0, 1})

    def test_deferred_acceptance_capacity(self):
        # 3 candidates, 1 firm, capacity 2
        # All candidates prefer Firm 0
        # Firm 0 prefers C2 > C1 > C0

        candidate_prefs = np.array([
            [0],
            [0],
            [0]
        ])

        firm_prefs_rank = np.array([
            [2, 1, 0] # F0: C2=0, C1=1, C0=2
        ])

        firm_capacities = [2]

        matched = deferred_acceptance(candidate_prefs, firm_prefs_rank, firm_capacities)

        self.assertIn(1, matched)
        self.assertIn(2, matched)
        self.assertNotIn(0, matched)

if __name__ == '__main__':
    unittest.main()
