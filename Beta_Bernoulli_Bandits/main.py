import numpy as np
from simulation import Arm, Simulation

def run_experiment(n_agents, n_arms, n_rounds, n_simulations, n_0, output_file):
    settings = ['monoculture', 'monoculture_averaged', 'monoculture_ucb', 'polyculture-fixed', 'polyculture-random']
    total_regrets = {setting: [] for setting in settings}
    total_misclassified = {setting: [] for setting in settings}

    with open(output_file, 'a') as f:
        f.write(f"--- Starting Experiment: n={n_agents} ---\\n")

    for i in range(n_simulations):
        print(f"Running simulation {i+1}/{n_simulations} for n={n_agents}...")

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

    with open(output_file, 'a') as f:
        f.write(f"\\n--- Final Averages for n={n_agents} after {n_simulations} runs ---\\n")
        for setting in sorted(total_regrets.keys()):
            avg_regret = np.mean(total_regrets[setting])
            avg_misclassified = np.mean(total_misclassified[setting])
            f.write(f"{setting}: Regret={avg_regret:.2f}, Misclassified={avg_misclassified:.2f}\\n")
        f.write("\\n")

def main():
    n_arms = 100
    n_rounds = 100
    n_simulations = 50
    n_0 = 5
    output_file = "simulation_results.txt"

    with open(output_file, 'w') as f:
        pass

    run_experiment(n_agents=10, n_arms=n_arms, n_rounds=n_rounds,
                   n_simulations=n_simulations, n_0=n_0, output_file=output_file)

    run_experiment(n_agents=50, n_arms=n_arms, n_rounds=n_rounds,
                   n_simulations=n_simulations, n_0=n_0, output_file=output_file)

    run_experiment(n_agents=90, n_arms=n_arms, n_rounds=n_rounds,
                   n_simulations=n_simulations, n_0=n_0, output_file=output_file)

    print("All simulations complete.")

if __name__ == "__main__":
    main()
