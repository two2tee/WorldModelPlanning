# Fetch iteration models
# Run tests
# Update Iterative stats
# Update Tensorboard
import json
import os
import pickle

import torch
import re

from mdrnn.iteration_stats.iteration_result import IterationResult
from tests.test_suite_factory import get_planning_tester
from environment.environment_factory import get_environment
from os.path import exists, join
from vae.vae import VAE
from vae.vae_trainer import VaeTrainer
from mdrnn.mdrnn import MDRNN
from utility.preprocessor import Preprocessor
from utility.tensorboard_handler import TensorboardHandler
from planning.simulation.mcts_simulation import MCTS as MCTS_simulation
from planning.simulation.rolling_horizon_simulation import RHEA as RHEA_simulation
from planning.simulation.random_mutation_hill_climbing_simulation import RMHC as RMHC_simulation

with open('config.json') as config_file:
    config = json.load(config_file)

iteration_stats_dir = join(config['mdrnn_dir'], 'iteration_stats')


def reload_model(file_location):
    mdrnn = MDRNN(num_actions=environment.action_sampler.num_actions,
                  latent_size=config['latent_size'],
                  num_gaussians=config['mdrnn']['num_gaussians'],
                  num_hidden_units=config['mdrnn']['hidden_units'])

    if not exists(file_location):
        raise Exception('No MDRNN model found...')
    state = torch.load(file_location, map_location=torch.device('cpu'))
    mdrnn.load_state_dict(state['state_dict'])
    print(f'Reloaded MDRNN model')
    return mdrnn


def get_planning_agent():
    simulated_agents = {"RHEA": RHEA_simulation(*config['planning']['rolling_horizon'].values()),
                        "RMHC": RMHC_simulation(*config['planning']['random_mutation_hill_climb'].values()),
                        "MCTS": MCTS_simulation(*config['planning']['monte_carlo_tree_search'].values())}

    return simulated_agents[config['planning']['planning_agent']]


def get_digit_from_path(path):
    match = re.search(r'(\d+)', path)
    return int(match.group(0))


def log_iteration_test_results(iteration_result, experiment_name):
    logger = TensorboardHandler(is_logging=True)
    logger.start_log(name=f'{experiment_name}_iterative_planning_test_results')

    title = f'Iterative_Planning_Tests_Results_{iteration_result.test_name}_trials_{iteration_result.total_trials}'
    logger.log_iteration_max_reward(name=title,
                                         iteration=iteration_result.iteration, max_reward=iteration_result.get_average_max_reward())
    logger.log_iteration_avg_reward(name=title,
                                    iteration=iteration_result.iteration, avg_reward=iteration_result.get_average_total_reward())
    logger.end_log()

def save_iteration_stats(iteration_results, experiment_name):
    stats_filename = f'iterative_stats_{experiment_name}'
    stats_filepath = join(join(config['mdrnn_dir'], 'iteration_stats'), f'{stats_filename}')

    encoded_iteration_results = [iteration_result.to_dict() for iteration_result in list(iteration_results.values())
                                 if iteration_result.test_name]  # Only save completed iteration

    file_content = {'iteration_results': encoded_iteration_results}
    with open(f'{stats_filepath}.pickle', 'wb') as file:
        pickle.dump(file_content, file, protocol=pickle.HIGHEST_PROTOCOL)

def load_iteration_stats(experiment_name):
    stats_filename = f'iterative_stats_{experiment_name}'
    stats_filepath = join(iteration_stats_dir, f'{stats_filename}.pickle')
    mdrnn_filename = f"{config['mdrnn_dir']}/checkpoints/iterative_{config['experiment_name']}_{config['mdrnn_trainer']['mdrnn_best_filename']}"

    if exists(stats_filepath) and exists(mdrnn_filename):
        with open(f'{stats_filepath}', 'rb') as file:
            file_content = pickle.load(file)
            decoded_results = [IterationResult.to_obj(encoded_result) for encoded_result in file_content['iteration_results']]
            iteration_results = {}
            for iteration_result in decoded_results:
                iteration_results[iteration_result.iteration] = iteration_result
            return iteration_results
    return {}

experiment_names = ['World_Model_A', 'World_Model_B', 'World_Model_C', 'World_Model_D']
for experiment_name in experiment_names:
    frame_preprocessor = Preprocessor(config['preprocessor'])

    environment = get_environment(config)
    agent = get_planning_agent()

    vae_trainer = VaeTrainer(config, frame_preprocessor, TensorboardHandler(is_logging=True))
    vae = VAE(config)
    vae = vae_trainer.reload_model(vae)

    mdrnn_models_location = 'mdrnn/checkpoints/backups'
    files = [os.path.join(root, name) for root, dirs, files in os.walk(mdrnn_models_location) for name in files]
    files = [file for file in files if experiment_name in file]
    files.sort(key=lambda path: get_digit_from_path(path))
    print(files)

    iteration_results = load_iteration_stats(experiment_name)
    for file in files:
        current_iteration = int(get_digit_from_path(file))
        iteration_result = iteration_results[current_iteration]
        mdrnn = reload_model(file)
        tester = get_planning_tester(config, vae, mdrnn, frame_preprocessor, environment, agent)
        test_name, trial_actions, trials_rewards, trial_elites, trial_max_rewards, trial_seeds = tester.run_specific_test(config['iterative_trainer']['test_scenario'])

        iteration_result.test_name = test_name
        iteration_result.trial_seeds = trial_seeds
        iteration_result.total_trials = len(trial_max_rewards)
        iteration_result.trials_rewards = trials_rewards
        iteration_result.trials_max_rewards = trial_max_rewards
        iteration_results[current_iteration] = iteration_result
        log_iteration_test_results(iteration_result, experiment_name)
        save_iteration_stats(iteration_results, experiment_name)


