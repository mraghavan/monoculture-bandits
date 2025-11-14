import json
import numpy as np
import os
from datetime import datetime
from src.simulation import simulate_monoculture, simulate_polyculture
from tqdm import tqdm

def run_simulation(simulation_params):
    num_trials = simulation_params['num_trials']
    num_steps = simulation_params['num_steps']
    num_arms = simulation_params['num_arms']
    N0_values = simulation_params['N0_values']

    results = {}

    for N0 in N0_values:
        results[N0] = {}

        # Monoculture simulation
        outcomes = [simulate_monoculture(num_steps, N0, num_arms) for _ in tqdm(range(num_trials), desc=f"Monoculture N0={N0}")]
        failure_rate = 1 - np.mean(outcomes)
        std_err = np.std(outcomes) / np.sqrt(num_trials)
        results[N0]['monoculture'] = {'failure_rate': failure_rate, 'std_err': std_err}

        # Polyculture simulations
        for k in range(2, num_arms + 1):
            outcomes = [simulate_polyculture(k, num_steps, N0, num_arms) for _ in tqdm(range(num_trials), desc=f"Polyculture k={k} N0={N0}")]
            failure_rate = 1 - np.mean(outcomes)
            std_err = np.std(outcomes) / np.sqrt(num_trials)
            results[N0][f'polyculture_{k}'] = {'failure_rate': failure_rate, 'std_err': std_err}

    return results

def main():
    # Reduced num_trials for quick testing
    simulation_params = {
        'num_trials': 10,
        'num_steps': 1000,
        'num_arms': 4,
        'N0_values': [1, 5, 10]
    }

    # Run the simulation
    simulation_results = run_simulation(simulation_params)

    # Create a unique filename for the results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"results/simulation_results_{timestamp}.json"

    # Save the results to a JSON file
    with open(filename, 'w') as f:
        json.dump(simulation_results, f, indent=4)

    print(f"Simulation complete. Results saved to {filename}")

if __name__ == "__main__":
    main()
