import unittest

import numpy as np

from simulations.failure_rates.run import build_trial_seeds, prepare_trial_inputs
from src.simulation import simulate_monoculture, simulate_polyculture


class TestFailureRatesSeeding(unittest.TestCase):
    def test_build_trial_seeds_is_deterministic(self):
        seeds_a = build_trial_seeds(base_seed=42, num_trials=4)
        seeds_b = build_trial_seeds(base_seed=42, num_trials=4)

        np.testing.assert_array_equal(seeds_a, seeds_b)

    def test_build_trial_seeds_is_shared_across_grid(self):
        seeds_a = build_trial_seeds(base_seed=42, num_trials=4)
        seeds_b = build_trial_seeds(base_seed=42, num_trials=6)[:4]

        np.testing.assert_array_equal(seeds_a, seeds_b)

    def test_trial_inputs_share_means_across_conditions(self):
        trial_seed = 12345
        p_arms_a, rng_a = prepare_trial_inputs(trial_seed, num_arms=4, condition_index=0, num_conditions=4)
        p_arms_b, rng_b = prepare_trial_inputs(trial_seed, num_arms=4, condition_index=2, num_conditions=4)

        np.testing.assert_allclose(p_arms_a, p_arms_b)
        self.assertNotEqual(rng_a.bit_generator.state, rng_b.bit_generator.state)

    def test_simulations_replay_with_fixed_means_and_seed(self):
        p_arms = np.array([0.15, 0.35, 0.55, 0.75])

        mono_a = simulate_monoculture(20, 3, 4, p_arms=p_arms, rng=np.random.default_rng(101))
        mono_b = simulate_monoculture(20, 3, 4, p_arms=p_arms, rng=np.random.default_rng(101))
        poly_a = simulate_polyculture(2, 20, 3, 4, p_arms=p_arms, rng=np.random.default_rng(202))
        poly_b = simulate_polyculture(2, 20, 3, 4, p_arms=p_arms, rng=np.random.default_rng(202))

        self.assertEqual(mono_a, mono_b)
        self.assertEqual(poly_a, poly_b)


if __name__ == '__main__':
    unittest.main()
