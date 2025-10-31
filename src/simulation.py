from src.bandit import Bandit
from src.agent import Agent
import numpy as np

def simulate_monoculture(bandit, num_steps, N0):
    """
    Simulates a "monoculture" scenario where each agent observes the choices
    of all previous agents. This is equivalent to a single agent pulling the
    arm `num_steps` times.
    """
    agent = Agent(N0=N0, bandit=bandit)
    for _ in range(num_steps):
        arm = agent.choose_arm()
        reward = bandit.pull(arm)
        agent.update_belief(arm, reward)
    return agent.choose_arm() == bandit.best_arm

def simulate_polyculture(bandit, num_agents, num_steps, N0):
    """
    Simulates a "polyculture" scenario where each agent acts independently
    from a common starting point. Success is determined by an "all-seeing"
    agent that aggregates their final beliefs.
    """
    # First, establish a common prior belief based on N0 initial samples.
    progenitor_agent = Agent(N0=N0, bandit=bandit)
    common_prior = progenitor_agent.beliefs.copy()

    # Create agents who all start with a copy of the common prior.
    agents = []
    for _ in range(num_agents):
        agent = Agent(num_arms=len(bandit.p_arms))
        agent.beliefs = common_prior.copy()
        agents.append(agent)

    for step in range(num_steps):
        agent_to_act = agents[step % num_agents]
        arm = agent_to_act.choose_arm()
        reward = bandit.pull(arm)
        agent_to_act.update_belief(arm, reward)


    if num_agents == 0:
        return False

    omniscient_agent = Agent(num_arms=len(bandit.p_arms))
    aggregated_beliefs = np.sum([agent.beliefs for agent in agents], axis=0)

    # The omniscient agent's belief should be the common starting belief plus
    # the sum of all individual updates. We correct the aggregated sum by
    # subtracting the (N-1) extra copies of the common prior.
    correction = (num_agents - 1) * common_prior
    omniscient_agent.beliefs = aggregated_beliefs - correction

    final_choice = omniscient_agent.choose_arm()
    return final_choice == bandit.best_arm
