import numpy as np
from simulation import Arm, Simulation

def main():
    n_agents = 90
    n_arms = 100
    n_rounds = 100
    n_simulations = 50
    n_0 = 5

    settings = ['monoculture', 'polyculture-fixed', 'polyculture-random']
    total_regrets = {setting: [] for setting in settings}

    for i in range(n_simulations):
        print(f"Running simulation {i+1}/{n_simulations}...")

        # Hold the set of arms fixed for each setting in a single run
        arms = [Arm() for _ in range(n_arms)]

        for setting in settings:
            sim = Simulation(n_agents, n_arms, n_rounds, setting, arms, n_0=n_0)
            sim.run()
            regret = sim.calculate_bayesian_regret()
            total_regrets[setting].append(regret)

    print("\\n--- Average Bayesian Regret ---")
    for setting in settings:
        avg_regret = np.mean(total_regrets[setting])
        print(f"{setting}: {avg_regret:.2f}")

if __name__ == "__main__":
    main()
