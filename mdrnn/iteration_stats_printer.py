#  Copyright (c) 2020, - All Rights Reserved
#  This file is part of the Evolutionary Planning on a Learned World Model thesis.
#  Unauthorized copying of this file, via any medium is strictly prohibited without the consensus of the authors.
#  Written by Thor V.A.N. Olesen <thorolesen@gmail.com> & Dennis T.T. Nguyen <dennisnguyen3000@yahoo.dk>.

import pickle
from os.path import exists
from iteration_stats.iteration_result import IterationResult
from utility.tensorboard_handler import TensorboardHandler

def log_iteration_test_results(iteration_result, experiment_name):
    logger = TensorboardHandler(is_logging=True)

    logger.start_log(name=f'{experiment_name}_iterative_planning_test_results')
    title = f'Iterative_Planning_Tests_Results_{iteration_result.test_name}_trials_{iteration_result.total_trials}'

    logger.log_iteration_max_reward(name=title,
                                         iteration=iteration_result.iteration, max_reward=iteration_result.get_average_max_reward())
    logger.log_iteration_avg_reward(name=title,
                                    iteration=iteration_result.iteration, avg_reward=iteration_result.get_average_total_reward())
    logger.end_log()
    print(f'logged iteration {iteration_result.iteration} - {logger.log_dir_root}/{experiment_name}_iterative_planning_test_results')

experiment_name = 'World_Model_E'
filename = f'iteration_stats/iterative_stats_{experiment_name}.pickle'
if exists(filename):
    with open(f'{filename}', 'rb') as file:
        file_content = pickle.load(file)
        iteration_stats = [IterationResult.to_obj(encoded_result) for encoded_result in file_content['iteration_results']]
        [print(iteration_stat) for iteration_stat in iteration_stats]

        # for iteration_stat in iteration_stats:
        #     log_iteration_test_results(iteration_stat, experiment_name)

else:
    print(f'File not found: {filename}')


