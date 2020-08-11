""" Custom file loader for NTBEA file stats """
#  Copyright (c) 2020, - All Rights Reserved
#  This file is part of the Evolutionary Planning on a Learned World Model thesis.
#  Unauthorized copying of this file, via any medium is strictly prohibited without the consensus of the authors.
#  Written by Thor V.A.N. Olesen <thorolesen@gmail.com> & Dennis T.T. Nguyen <dennisnguyen3000@yahoo.dk>.

import pprint
import dill as pickle
from os.path import join, exists


def load_session(filename):
    session_dir = 'ntbea_sessions/backup'
    session_filename = filename
    session_filepath = join(session_dir, f'{session_filename}.pickle')

    if exists(session_filepath):
        with open(f'{session_filepath}', 'rb') as file:
            save_data = pickle.load(file)
            world_model = save_data['world_model']
            agent_type = save_data['agent_type']
            # self._load_data(save_data['ntbea_data'])
            search_space = save_data['search_space_data']
            # self._mutator.load_data(save_data['mutate_data'])
            # self._tuple_landscape.load_data(save_data['tuple_landscape_data'])

            return save_data['iterations'], save_data['best_solution'], save_data['best_fitness'], save_data['tuple_landscape_data'],\
                   search_space, world_model, agent_type
    raise Exception('File not found')

def print_results(best_params_config, fitness, agent_type, world_model, iterations):
    print(f'COMPLETED {iterations} NTBEA Iterations for planning agent: {agent_type} with world model: {world_model}')
    stats = f'-- Best Configuration --' \
            f'\nFitness: {fitness}' \
            f'\nShift buffer: {"True" if best_params_config[0]== 1 else "False"}' \
            f'\nHorizon: {best_params_config[1]}' \
            f'\nGenerations: {best_params_config[2]}'\
            f'\nIs rollout: {best_params_config[3]}' \
            f'\nMax rollouts : {best_params_config[4]}' \
            f'\nMutation Method: {best_params_config[5]}'

    stats = stats+f'\nPopulation: {best_params_config[6]}' \
                  f'\nGenetic Operator: {best_params_config[7]}' \
                  f'\nSelection Method: {best_params_config[8]}' \
                  f'\nCrossover Method: {best_params_config[9]}' \
                  if agent_type == "RHEA" else stats
    print(stats)


filename = 'ntbea_session_World_Model_M_RMHC_100'
iterations, solution, fitness, tuples, search_space, world_model, agent_type = load_session(filename)

print_results(solution, fitness, agent_type, world_model, iterations)

print(search_space)
pp = pprint.PrettyPrinter(indent=0)
one_tuple = [(i,) for i in range(len(search_space['search_space']))]
one_tuple_stats = [tuples['tuple_stats'][key] for key in one_tuple]
pp.pprint(one_tuple_stats)
