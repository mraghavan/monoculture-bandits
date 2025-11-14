import json
import numpy as np
import os
from src.simulation import simulate_monoculture, simulate_polyculture
from tqdm import tqdm

def run_and_save_experiment(params):
    """
    Runs a single simulation experiment and saves the results to a JSON file.
    """
    num_trials = params['num_trials']
    num_steps = params['num_steps']
    num_arms = params['num_arms']
    N0 = params['N0']
    condition = params['condition']
    num_agents = params.get('num_agents', 1)  # Default to 1 for monoculture

    # Run the appropriate simulation
    if condition == 'monoculture':
        outcomes = [simulate_monoculture(num_steps, N0, num_arms) for _ in tqdm(range(num_trials), desc=f"Monoculture N0={N0}")]
    elif condition == 'polyculture':
        outcomes = [simulate_polyculture(num_agents, num_steps, N0, num_arms) for _ in tqdm(range(num_trials), desc=f"Polyculture k={num_agents} N0={N0}")]
    else:
        raise ValueError(f"Unknown condition: {condition}")

    # Calculate results
    failure_rate = 1 - np.mean(outcomes)
    std_err = np.std(outcomes) / np.sqrt(num_trials)

    result_data = {
        'params': params,
        'results': {
            'failure_rate': failure_rate,
            'std_err': std_err
        }
    }

    # Ensure the results directory exists
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Create a descriptive filename
    if condition == 'monoculture':
        filename = f"N0_{N0}_arms_{num_arms}_steps_{num_steps}_mono.json"
    else:
        filename = f"N0_{N0}_arms_{num_arms}_steps_{num_steps}_poly_{num_agents}.json"

    filepath = os.path.join(results_dir, filename)

    # Save the results
    with open(filepath, 'w') as f:
        json.dump(result_data, f, indent=4)

    print(f"Saved results to {filepath}")

def main():
    """
    Defines and runs a suite of simulation experiments.
    """
    base_params = {
        'num_trials': 10,  # Reduced for quick testing
        'num_steps': 1000,
        'num_arms': 4,
    }
    N0_values = [1, 5, 10]

    for N0 in N0_values:
        # Monoculture simulation
        mono_params = base_params.copy()
        mono_params.update({
            'N0': N0,
            'condition': 'monoculture',
        })
        run_and_save_experiment(mono_params)

        # Polyculture simulations
        for k in range(2, base_params['num_arms'] + 1):
            poly_params = base_params.copy()
            poly_params.update({
                'N0': N0,
                'condition': 'polyculture',
                'num_agents': k
            })
            run_and_save_experiment(poly_params)

    print("All simulations complete.")

if __name__ == "__main__":
    main()
