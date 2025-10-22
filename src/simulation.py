from src.bandit import Bandit
from src.agent import Agent
import numpy as np

def simulate_single_agent(num_steps, N0):
    """
    Simulates a single agent interacting with a bandit for a fixed number of steps.
    """
    bandit = Bandit()
    agent = Agent(N0=N0)
    for _ in range(num_steps):
        arm = agent.choose_arm()
        reward = bandit.pull(arm)
        agent.update_belief(arm, reward)
    return agent.choose_arm() == bandit.best_arm

def simulate_independent_agents(num_agents, num_steps, N0):
    """
    Simulates multiple agents with independent priors who do not share information.
    The total number of pulls is `num_steps`, distributed among the agents.
    """
    bandit = Bandit()
    # Each agent gets its own randomly drawn prior
    agents = [Agent(N0=N0) for _ in range(num_agents)]

    for step in range(num_steps):
        # Cycle through the agents to distribute the arm pulls
        agent_to_act = agents[step % num_agents]
        arm = agent_to_act.choose_arm()
        reward = bandit.pull(arm)

        # Only the agent who acted updates its belief
        agent_to_act.update_belief(arm, reward)

    # Success is determined by an "all-seeing" agent who observes the
    # entire reward history.
    if num_agents == 0:
        return False

    # We simulate this by creating a new agent and giving it the aggregated
    # beliefs of all independent agents. The N0 for this agent's prior is
    # not used since we overwrite the beliefs, but we pass it for consistency.
    omniscient_agent = Agent(num_arms=len(bandit.p_arms), N0=N0)

    # Each agent's belief is its prior + its observations. Summing them up is
    # like one agent with a summed prior who saw all observations.
    aggregated_beliefs = np.sum([agent.beliefs for agent in agents], axis=0)
    omniscient_agent.beliefs = aggregated_beliefs

    final_choice = omniscient_agent.choose_arm()
    return final_choice == bandit.best_arm
