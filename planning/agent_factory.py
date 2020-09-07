from planning.simulation.mcts_simulation import MCTS as MCTS_simulation
from planning.simulation.rolling_horizon_simulation import RHEA as RHEA_simulation
from planning.simulation.random_mutation_hill_climbing_simulation import RMHC as RMHC_simulation
from planning.simulation.random_simulation import RandomAgent as RANDOM_simulation


def get_planning_agent(config):
    agent = config['planning']['planning_agent']
    print(agent)
    return _get_planning_agent(agent, config)


def _get_planning_agent(agent, config):
    simulated_agents = {"RHEA": RHEA_simulation(*config['planning']['rolling_horizon'].values()),
                        "RMHC": RMHC_simulation(*config['planning']['random_mutation_hill_climb'].values()),
                        "MCTS": MCTS_simulation(*config['planning']['monte_carlo_tree_search'].values()),
                        "RANDOM": RANDOM_simulation()
                        }
    if agent not in simulated_agents:
        raise Exception(f'Invalid agent type: {agent} - available agents: {list(simulated_agents.keys())} ')

    return simulated_agents[agent]


def get_agent_parameters(config):
    agent = config['planning']['planning_agent']
    agent_params = {
        'RHEA': config['planning']['rolling_horizon'],
        'RMHC': config['planning']['random_mutation_hill_climb'],
        "MCTS": config['planning']['monte_carlo_tree_search'],
        "RANDOM": {}
    }
    if agent not in agent_params:
        raise Exception(f'Invalid agent type: {agent} - available agents: {list(agent_params.keys())} ')

    return agent_params[agent]
