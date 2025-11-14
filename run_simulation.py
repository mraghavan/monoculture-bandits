import json
import numpy as np
import os
from src.simulation import simulate_monoculture, simulate_polyculture
from tqdm import tqdm

def run_experiment(params):
    """
    Runs a single simulation experiment and returns the results.
    """
    num_trials = params['num_trials']
    num_steps = params['num_steps']
    num_arms = params['num_arms']
    N0 = params['N0']
    condition = params['condition']
    num_agents = params.get('num_agents', 1)  # Default to 1 for monoculture

    # Run the appropriate simulation
    if condition == 'monoculture':
        desc = f"Monoculture N0={N0}, steps={num_steps}"
        outcomes = [simulate_monoculture(num_steps, N0, num_arms) for _ in tqdm(range(num_trials), desc=desc)]
    elif condition == 'polyculture':
        desc = f"Polyculture k={num_agents} N0={N0}, steps={num_steps}"
        outcomes = [simulate_polyculture(num_agents, num_steps, N0, num_arms) for _ in tqdm(range(num_trials), desc=desc)]
    else:
        raise ValueError(f"Unknown condition: {condition}")

    # Calculate results
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
        'num_trials': 10,  # Reduced for quick testing
        'num_arms': 4,
    }
    N0_values = [1, 5, 10]
    num_steps_values = [1000, 2000]

    for num_steps in num_steps_values:
        results_list = []
        for N0 in N0_values:
            # Monoculture simulation
            mono_params = base_params.copy()
            mono_params.update({
                'N0': N0,
                'num_steps': num_steps,
                'condition': 'monoculture',
            })
            results_list.append(run_experiment(mono_params))

            # Polyculture simulations
            for k in range(2, base_params['num_arms'] + 1):
                poly_params = base_params.copy()
                poly_params.update({
                    'N0': N0,
                    'num_steps': num_steps,
                    'condition': 'polyculture',
                    'num_agents': k
                })
                results_list.append(run_experiment(poly_params))

        # Save all results for this num_steps value to a single file
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
