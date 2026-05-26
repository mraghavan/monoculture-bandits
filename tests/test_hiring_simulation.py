import unittest
from hiring_crowd_wisdom import deferred_acceptance

class TestDeferredAcceptance(unittest.TestCase):

    def test_simple_stable_matching(self):
        """
        Test a simple case where a stable matching is known.
        2 firms, 3 candidates, capacity of 1 for each firm.
        """
        # Firm preferences: firm_id -> list of candidate_ids in order of preference
        firm_prefs = {
            0: [0, 1, 2],  # Firm 0 prefers C0 > C1 > C2
            1: [1, 0, 2]   # Firm 1 prefers C1 > C0 > C2
        }

        # Candidate preferences: candidate_id -> list of firm_ids in order of preference
        candidate_prefs = {
            0: [0, 1],     # Candidate 0 prefers F0 > F1
            1: [1, 0],     # Candidate 1 prefers F1 > F0
            2: [0, 1]      # Candidate 2 prefers F0 > F1
        }

        firm_capacities = {
            0: 1,
            1: 1
        }

        # Expected matching: F0 gets C0, F1 gets C1. C2 is unmatched.
        # This is stable because:
        # - F0 is matched with its top choice C0.
        # - F1 is matched with its top choice C1.
        # - C0 and C1 are matched with their top choices.
        # - C2 would prefer F0, but F0 prefers its current match C0 over C2.
        # - C2 would prefer F1, but F1 prefers its current match C1 over C2.
        expected_matches = {
            0: [0],
            1: [1]
        }

        result = deferred_acceptance(firm_prefs, candidate_prefs, firm_capacities)

        # The algorithm may return lists in different orders, so we sort them.
        sorted_result = {k: sorted(v) for k, v in result.items()}
        sorted_expected = {k: sorted(v) for k, v in expected_matches.items()}

        self.assertEqual(sorted_result, sorted_expected)

    def test_multiple_capacity(self):
        """
        Test a case with firm capacity greater than 1.
        """
        # 2 firms, 4 candidates
        firm_prefs = {
            0: [0, 1, 2, 3], # F0: C0 > C1 > C2 > C3
            1: [3, 2, 1, 0]  # F1: C3 > C2 > C1 > C0
        }

        candidate_prefs = {
            0: [0, 1], # C0: F0 > F1
            1: [0, 1], # C1: F0 > F1
            2: [1, 0], # C2: F1 > F0
            3: [1, 0]  # C3: F1 > F0
        }

        firm_capacities = {
            0: 2,
            1: 2
        }

        # Expected result: C0, C1 go to F0. C2, C3 go to F1.
        expected_matches = {
            0: [0, 1],
            1: [2, 3]
        }

        result = deferred_acceptance(firm_prefs, candidate_prefs, firm_capacities)
        sorted_result = {k: sorted(v) for k, v in result.items()}
        sorted_expected = {k: sorted(v) for k, v in expected_matches.items()}
        self.assertEqual(sorted_result, sorted_expected)

if __name__ == '__main__':
    unittest.main()
