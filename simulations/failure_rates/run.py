import json
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.simulation import simulate_monoculture, simulate_polyculture
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial

BASE_SEED = 42

# Prior distribution parameters for bandit arm probabilities
# Arm probabilities are drawn from Beta(alpha, beta)
BANDIT_PRIOR_ALPHA = 2
BANDIT_PRIOR_BETA = 2

def build_trial_seeds(base_seed, num_trials):
    seed_sequence = np.random.SeedSequence(base_seed)
    seed_rng = np.random.default_rng(seed_sequence)
    return seed_rng.integers(0, 2**32, size=num_trials, dtype=np.uint32)

def prepare_trial_inputs(trial_seed, num_arms, condition_index, num_conditions):
    seed_sequence = np.random.SeedSequence(int(trial_seed))
    child_sequences = seed_sequence.spawn(1 + num_conditions)

    shared_rng = np.random.default_rng(child_sequences[0])
    p_arms = shared_rng.beta(BANDIT_PRIOR_ALPHA, BANDIT_PRIOR_BETA, size=num_arms)
    condition_rng = np.random.default_rng(child_sequences[1 + condition_index])
    return p_arms, condition_rng

# Wrapper functions for parallel execution. They must be at the top level.
def run_mono_trial(num_steps, N0, num_arms, condition_index, num_conditions, trial_seed):
    p_arms, rng = prepare_trial_inputs(trial_seed, num_arms, condition_index, num_conditions)
    return simulate_monoculture(num_steps, N0, num_arms, p_arms=p_arms, rng=rng)

def run_poly_trial(num_agents, num_steps, N0, num_arms, condition_index, num_conditions, trial_seed):
    p_arms, rng = prepare_trial_inputs(trial_seed, num_arms, condition_index, num_conditions)
    return simulate_polyculture(num_agents, num_steps, N0, num_arms, p_arms=p_arms, rng=rng)

def run_experiment(params, trial_seeds):
    """
    Runs a single simulation experiment in parallel and returns the results.
    """
    num_trials = params['num_trials']
    num_steps = params['num_steps']
    num_arms = params['num_arms']
    N0 = params['N0']
    condition = params['condition']
    num_agents = params.get('num_agents', 1)
    num_conditions = num_arms

    if condition == 'monoculture':
        desc = f"Monoculture N0={N0}, steps={num_steps}"
        condition_index = 0
        target_func = partial(run_mono_trial, num_steps, N0, num_arms, condition_index, num_conditions)
    elif condition == 'polyculture':
        desc = f"Polyculture k={num_agents} N0={N0}, steps={num_steps}"
        condition_index = num_agents - 1
        target_func = partial(
            run_poly_trial,
            num_agents,
            num_steps,
            N0,
            num_arms,
            condition_index,
            num_conditions,
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")

    with ProcessPoolExecutor() as executor:
        outcomes = list(tqdm(executor.map(target_func, trial_seeds), total=num_trials, desc=desc))

    failure_rate = 1 - np.mean(outcomes)
    std_err = np.std(outcomes) / np.sqrt(num_trials)

    return {
        'params': params,
        'results': {
            'failure_rate': failure_rate,
            'std_err': std_err
        }
    }

def main():
    """
    Defines and runs a suite of simulation experiments.
    """
    base_params = {
        'num_trials': 10000,  # Increased trials to see performance gain
        'num_arms': 4,
        'bandit_prior_alpha': BANDIT_PRIOR_ALPHA,  # Beta distribution alpha for generating arm probabilities
        'bandit_prior_beta': BANDIT_PRIOR_BETA,   # Beta distribution beta for generating arm probabilities
    }
    N0_values = [1, 3, 5, 8, 10, 15, 20, 25, 35, 50, 75, 100]
    num_steps_values = [100, 1000]

    for num_steps in num_steps_values:
        results_list = []
        for N0 in N0_values:
            trial_seeds = build_trial_seeds(BASE_SEED, base_params['num_trials'])
            mono_params = base_params.copy()
            mono_params.update({
                'N0': N0,
                'num_steps': num_steps,
                'condition': 'monoculture',
            })
            results_list.append(run_experiment(mono_params, trial_seeds))

            for k in range(2, base_params['num_arms'] + 1):
                poly_params = base_params.copy()
                poly_params.update({
                    'N0': N0,
                    'num_steps': num_steps,
                    'condition': 'polyculture',
                    'num_agents': k
                })
                results_list.append(run_experiment(poly_params, trial_seeds))

        results_dir = 'results'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        filename = f"arms_{base_params['num_arms']}_steps_{num_steps}.json"
        filepath = os.path.join(results_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(results_list, f, indent=4)

        print(f"Saved results for num_steps={num_steps} to {filepath}")

    print("All simulations complete.")

if __name__ == "__main__":
    main()
