import numpy as np
from simulation import Arm, Simulation
import time

def main():
    n_agents = 90
    n_arms = 100
    n_rounds = 100
    n_simulations = 100
    n_0 = 5

    settings = ['monoculture', 'monoculture_informed', 'monoculture_averaged',
                'monoculture_maxed', 'monoculture_minned', 'polyculture-fixed',
                'polyculture-random']
    total_regrets = {setting: [] for setting in settings}
    total_misclassified = {setting: [] for setting in settings}

    completed_simulations = 0
    output_file = "simulation_results.txt"

    with open(output_file, 'w') as f:
        pass

    for i in range(n_simulations):
        print(f"Running simulation {i+1}/{n_simulations}...")

        arms = [Arm() for _ in range(n_arms)]

        initial_samples = {}
        for arm_idx in range(n_arms):
            initial_samples[arm_idx] = []
            for agent_idx in range(n_agents):
                samples = [arms[arm_idx].pull() for _ in range(n_0)]
                initial_samples[arm_idx].append(samples)

        for setting in settings:
            sim = Simulation(n_agents, n_arms, n_rounds, setting, arms, initial_samples)
            sim.run()
            regret = sim.calculate_bayesian_regret()
            misclassified = sim.calculate_misclassified_arms()
            total_regrets[setting].append(regret)
            total_misclassified[setting].append(misclassified)

        completed_simulations += 1

        with open(output_file, 'a') as f:
            f.write(f"--- Results after {completed_simulations} runs ---\\n")
            for setting in sorted(total_regrets.keys()):
                avg_regret = np.mean(total_regrets[setting])
                avg_misclassified = np.mean(total_misclassified[setting])
                f.write(f"{setting}: Regret={avg_regret:.2f}, Misclassified={avg_misclassified:.2f}\\n")

    with open(output_file, 'a') as f:
        f.write(f"\\n--- Final Averages after {completed_simulations} runs ---\\n")
        for setting in sorted(total_regrets.keys()):
            avg_regret = np.mean(total_regrets[setting])
            avg_misclassified = np.mean(total_misclassified[setting])
            f.write(f"{setting}: Regret={avg_regret:.2f}, Misclassified={avg_misclassified:.2f}\\n")

if __name__ == "__main__":
    main()
