from src.bandit import Bandit
from src.agent import Agent
import numpy as np

def simulate_monoculture(num_agents, N0):
    """
    Simulates a "monoculture" scenario where each agent observes the choices
    of all previous agents. This is equivalent to a single agent pulling the
    arm `num_agents` times.
    """
    bandit = Bandit()
    agent = Agent(N0=N0, bandit=bandit)
    for _ in range(num_agents):
        arm = agent.choose_arm()
        reward = bandit.pull(arm)
        agent.update_belief(arm, reward)
    return agent.choose_arm() == bandit.best_arm

def simulate_polyculture(num_agents, N0):
    """
    Simulates a "polyculture" scenario where each agent acts independently.
    Success is determined by an "all-seeing" agent that aggregates their
    final beliefs.
    """
    bandit = Bandit()
    agents = [Agent(N0=N0, bandit=bandit) for _ in range(num_agents)]

    for agent in agents:
        arm = agent.choose_arm()
        reward = bandit.pull(arm)
        agent.update_belief(arm, reward)

    if num_agents == 0:
        return False

    omniscient_agent = Agent(num_arms=len(bandit.p_arms))
    aggregated_beliefs = np.sum([agent.beliefs for agent in agents], axis=0)
    omniscient_agent.beliefs = aggregated_beliefs

    final_choice = omniscient_agent.choose_arm()
    return final_choice == bandit.best_arm
