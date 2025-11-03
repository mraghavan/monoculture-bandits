import numpy as np
from simulation import Arm, Simulation

def main():
    n_agents = 90
    n_arms = 100
    n_rounds = 100
    n_simulations = 50
    n_0 = 5

    settings = ['monoculture', 'monoculture_informed', 'polyculture-fixed', 'polyculture-random']
    total_regrets = {setting: [] for setting in settings}

    for i in range(n_simulations):
        print(f"Running simulation {i+1}/{n_simulations}...")

        arms = [Arm() for _ in range(n_arms)]

        # Pre-generate all initial samples for all arms and all agents
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
            total_regrets[setting].append(regret)

    print("\n--- Average Bayesian Regret ---")
    for setting in settings:
        avg_regret = np.mean(total_regrets[setting])
        print(f"{setting}: {avg_regret:.2f}")

if __name__ == "__main__":
    main()
