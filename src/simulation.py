from src.bandit import Bandit
from src.agent import Agent
import numpy as np

def simulate_monoculture(num_steps, N0, num_arms):
    """
    Simulates a single agent interacting with a bandit for a fixed number of steps.
    """
    bandit = Bandit(num_arms=num_arms)
    agent = Agent(num_arms=num_arms, N0=N0, bandit=bandit)

    for _ in range(num_steps):
        arm = agent.choose_arm()
        reward = bandit.pull(arm)
        agent.update_belief(arm, reward)
    return agent.choose_arm() == bandit.best_arm

def simulate_polyculture(num_agents, num_steps, N0, num_arms):
    """
    Simulates multiple agents with independent priors who do not share information.
    The total number of pulls is `num_steps`, distributed among the agents.
    """
    bandit = Bandit(num_arms=num_arms)

    # Progenitor agent establishes a common prior
    progenitor = Agent(num_arms=num_arms, N0=N0, bandit=bandit)
    common_prior = progenitor.beliefs.copy() # Capture the prior after N0 pulls

    agents = [Agent(num_arms=num_arms, N0=0) for _ in range(num_agents)]
    for agent in agents:
        agent.beliefs = common_prior.copy()

    steps_per_agent = num_steps // num_agents
    for i in range(num_agents):
        for _ in range(steps_per_agent):
            arm = agents[i].choose_arm()
            reward = bandit.pull(arm)
            agents[i].update_belief(arm, reward)

    # Success is determined by an "all-seeing" agent who observes the
    # entire reward history.
    omniscient_agent = Agent(num_arms=num_arms, N0=0)

    # Aggregate beliefs
    aggregated_beliefs = np.sum([agent.beliefs for agent in agents], axis=0)

    # Correct for the common prior being counted multiple times
    correction = (num_agents - 1) * common_prior
    omniscient_agent.beliefs = aggregated_beliefs - correction

    final_choice = omniscient_agent.choose_arm()
    return final_choice == bandit.best_arm
