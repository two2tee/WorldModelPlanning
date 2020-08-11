""" NTBEA Implementation based on https://github.com/bam4d/NTBEA """
#  Copyright (c) 2020, - All Rights Reserved
#  This file is part of the Evolutionary Planning on a Learned World Model thesis.
#  Unauthorized copying of this file, via any medium is strictly prohibited without the consensus of the authors.
#  Written by Thor V.A.N. Olesen <thorolesen@gmail.com> & Dennis T.T. Nguyen <dennisnguyen3000@yahoo.dk>.

import os
import copy
import logging
import numpy as np
import dill as pickle
from tqdm import tqdm
from os.path import join, exists
from collections import defaultdict


class Mutator(object):
    '''
    Inherited by other mutator objects
    '''

    def __init__(self, name):
        self._name = name

    def get_name(self):
        return self._name

    def mutate(self, x):
        raise NotImplementedError()


class SearchSpace(object):
    '''
    Inherited by other search space objects
    '''

    def __init__(self, name, ndims):
        self._ndims = ndims
        self._name = name

    def get_name(self):
        return self._name

    def get_num_dims(self):
        return self._ndims

    def get_random_point(self):
        raise NotImplementedError()

    def get_size(self):
        raise NotImplementedError()

    def get_dim_size(self, j):
        raise NotImplementedError()

    def get_valid_values_in_dim(self, dim):
        raise NotImplementedError()


class Evaluator(object):
    '''
    Inherited by other evalutator objects
    '''

    def __init__(self, name):
        self._name = name

    def get_name(self):
        return self._name

    def evaluate(self, x):
        raise NotImplementedError()


class BanditLandscapeModel(object):

    def __init__(self, name):
        self._name = name

    def get_name(self):
        return self._name

    def reset(self):
        raise NotImplementedError()

    def init(self):
        raise NotImplementedError()

    def add_evaluated_point(self, point, fitness):
        raise NotImplementedError()

    def get_mean_estimtate(self, point):
        raise NotImplementedError()

    def get_exploration_estimate(self, point):
        raise NotImplementedError()

    def get_tuple_data(self):
        raise NotImplementedError()

    def load_tuple_data(self, data):
        raise NotImplementedError()


class NTupleLandscape(BanditLandscapeModel):
    '''
    The N-tuple landscape implementation
    '''

    def __init__(self, search_space, tuple_config=None, ucb_epsilon=0.5):
        super(NTupleLandscape, self).__init__('N-Tuple Bandit Landscape')
        self._logger = logging.getLogger('NTupleLandscape')

        # If we dont have a tuple config, we just create a default tuple config, the 1-tuples and N-tuples
        if tuple_config == None:
            tuple_config = [1, search_space.get_num_dims()]
        self._tuple_config = set(tuple_config)
        self._tuples = list()
        self._ndims = search_space.get_num_dims()
        self._sampled_points = set()
        self._ucb_epsilon = ucb_epsilon
        self._tuple_stats = None
        self.reset()

    def reset(self):
        self._tuple_stats = defaultdict(
            lambda: defaultdict(
                lambda: {
                    'n': 0,
                    'min': 0.0,
                    'max': 0.0,
                    'sum': 0.0,
                    'sum_squared': 0.0
                }
            )
        )

    def get_tuple_combinations(self, r, ndims):
        '''
        Get the unique combinations of tuples for the n-tuple landscape
        :param n: the 'n' value of this tuple
        :param ndims: the number of dimensions in the search space
        :return:
        '''
        return self._get_unique_combinations(0, r, range(0, ndims))

    def _get_unique_combinations(self, idx, r, source_array):

        result = []
        for i in range(idx, len(source_array)):
            if r - 1 > 0:
                next_level = self._get_unique_combinations(i + 1, r - 1, source_array)
                for x in next_level:
                    value = [source_array[i]]
                    value.extend(x)
                    result.append(value)

            else:
                result.append([source_array[i]])

        return result

    def init(self):
        '''
        Create the index combinations for each of the n-tules
        '''
        # Create all possible tuples for each
        for n in self._tuple_config:
            n_tuples = [tup for tup in self.get_tuple_combinations(n, self._ndims)]
            self._tuples.extend(n_tuples)
            self._logger.debug('Added %d-tuples: %s' % (n, n_tuples))

        self._logger.info('Tuple Landscape Size: %d' % len(self._tuples))

    def add_evaluated_point(self, point, fitness):
        """
        Add a point and it's fitness to the tuple landscape
        """

        self._sampled_points.add(tuple(point))

        for tup in self._tuples:
            # The search space value is the values given by applying the tuple to the search space.
            # This is used to index the stats at that point for the particular tuple in question
            search_space_value = point[tup]

            # Use 'totals' as a key to store summary data of the tuple
            self._tuple_stats[tuple(tup)]['totals']['n'] += 1

            search_space_tuple_stats = self._tuple_stats[tuple(tup)][tuple(search_space_value)]
            search_space_tuple_stats['n'] += 1
            search_space_tuple_stats['max'] = np.maximum(search_space_tuple_stats['max'], fitness)
            search_space_tuple_stats['min'] = np.minimum(search_space_tuple_stats['max'], fitness)
            search_space_tuple_stats['sum'] += fitness
            search_space_tuple_stats['sum_squared'] += fitness ** 2

    def get_mean_estimtate(self, point):
        """
        Iterate over all the tuple stats we have stored for this point and sum the means and the number
        of stats we have found.

        Finally the sum of the means divided by the total number of stats found is returned
        """
        sum = 0
        tuple_count = 0
        for tup in self._tuples:
            search_space_value = point[tup]
            tuple_stats = self._tuple_stats[tuple(tup)][tuple(search_space_value)]
            if tuple_stats['n'] > 0:
                sum += tuple_stats['sum'] / tuple_stats['n']
                tuple_count += 1

        if tuple_count == 0:
            return 0
        return sum / tuple_count

    def get_exploration_estimate(self, point):
        """
        Calculate the average of the exploration across all tuples of the exploration
        """

        sum = 0
        tuple_count = len(self._tuples)

        for tup in self._tuples:
            search_space_value = point[tup]
            tuple_stats = self._tuple_stats[tuple(tup)]
            search_space_tuple_stats = tuple_stats[tuple(search_space_value)]
            if search_space_tuple_stats['n'] == 0:
                n = tuple_stats['totals']['n']
                sum += np.sqrt(np.log(1 + n) / self._ucb_epsilon)
            else:
                n = search_space_tuple_stats['n']
                sum += np.sqrt(np.log(1 + n) / (n + self._ucb_epsilon))

        return sum / tuple_count

    def get_best_sampled(self):
        current_best_mean = 0
        current_best_point = None
        for point in self._sampled_points:
            mean = self.get_mean_estimtate(np.array(point))

            if mean > current_best_mean:
                current_best_mean = mean
                current_best_point = point

        return current_best_point

    def get_data(self):
        data = {'tuple_stats': self._tuple_stats,
                'tuple_config': self._tuple_config,
                'tuples': self._tuples,
                'ndims': self._ndims,
                'sampled_points': self._sampled_points}
        return data

    def load_data(self, data):
        self._tuple_stats = data['tuple_stats']
        self._tuple_config = data['tuple_config']
        self._tuples = data['tuples']
        self._ndims = data['ndims']
        self._sampled_points = data['sampled_points']

class NTupleEvolutionaryAlgorithm():

    def __init__(self, tuple_landscape, evaluator, search_space, mutator, k_explore=100, n_samples=1,
                 eval_neighbours=50, world_model=None, agent_type=None, config=None):
        self._logger = logging.getLogger('NTupleEvolutionaryAlgorithm')

        self._tuple_landscape = tuple_landscape
        self._evaluator = evaluator
        self._search_space = search_space
        self._mutator = mutator
        self._k_explore = k_explore
        self._n_samples = n_samples
        self._eval_neighbours = min(eval_neighbours, search_space.get_size())
        self._tie_break_noise = 1e-6

        self._logger.info('Search Space: %s' % self._search_space.get_name())
        self._logger.info('Evaluator: %s' % self._evaluator.get_name())
        self._logger.info('Mutator: %s' % self._mutator.get_name())
        self._world_model = world_model
        self._agent_type = agent_type
        self.config = config
        self.session_dir = 'tuning/ntbea_sessions'
        if not exists(self.session_dir):
            os.mkdir(self.session_dir)

    def _evaluate_landscape(self, point):

        self._logger.debug('Estimating landscape around %s', point)

        evaluated_neighbours = set()

        unique_neighbours = 0
        current_best_ucb = 0
        current_best_neighbour = None

        # Loop until we have the required numbers of estimated neighbours
        while unique_neighbours < self._eval_neighbours:
            potential_neighbour = self._mutator.mutate(point)

            # If we already have estimates for this unique neighbour then just get a new one
            if tuple(potential_neighbour) in evaluated_neighbours:
                continue

            unique_neighbours += 1
            exploit = self._tuple_landscape.get_mean_estimtate(potential_neighbour)
            explore = self._tuple_landscape.get_exploration_estimate(potential_neighbour)
            ucb_with_noise = exploit + self._k_explore * explore + np.random.uniform() * self._tie_break_noise
            evaluated_neighbours.add(tuple(potential_neighbour))
            self._logger.debug('Neighbour UCB %.2f at %s' % (ucb_with_noise, potential_neighbour))

            # Track the best neighbour that we have seen so far
            if current_best_ucb < ucb_with_noise:
                self._logger.debug('Found best UCB %.2f at %s' % (ucb_with_noise, potential_neighbour))
                print('Found best UCB %.2f at %s' % (ucb_with_noise, potential_neighbour))
                current_best_ucb = ucb_with_noise
                current_best_neighbour = copy.deepcopy(potential_neighbour)

        return current_best_neighbour

    def run(self, n_evaluations):
        start_eval = 0
        best_solution = None
        best_fitness = 0

        if self.config is not None and self.config['ntbea_tuning']['is_reload_session']:
            start_eval, best_solution, best_fitness = self._load_session()
            n_evaluations = n_evaluations if start_eval < n_evaluations else n_evaluations+start_eval

        print(f'Starting from iteration {start_eval} and run until {n_evaluations}')

        self._tuple_landscape.init()

        # Get a random point to start
        point = self._search_space.get_random_point()
        print(f'Initial Random configuration {point}')

        # Repeat many times
        progress_bar = tqdm(total=n_evaluations, desc="NTBEA Iteration",)
        progress_bar.set_postfix_str(f"Solution: {point}")

        for eval in range(start_eval, n_evaluations):
            print(f'\n-- NTBEA Iteration: {eval} --\n')
            # Explore the neighbourhood in the tuple landscape and find a strong next candidate point
            # Skip this if this is the first iteration of the algorithm because it will be a random point
            if eval > 0:
                point = self._evaluate_landscape(point)
                print(f'Current evaluated: {point}')

            # Evaluate the point (is repeated several times if n_samples > 0)
            fitness = np.mean([self._evaluator.evaluate(point) for _ in range(0, self._n_samples)])

            self._logger.debug('Evaluated fitness: %.2f at %s' % (fitness, point))

            # Add the new point to the tuple landscape
            self._tuple_landscape.add_evaluated_point(point, fitness)

            # Set initial best fitness for tracking
            if best_solution is None:
                best_solution = point
                best_fitness = fitness

            if eval % 10 == 0 and eval != 0:
                print(' --- Update and eval best point. ---')
                best_solution = self._tuple_landscape.get_best_sampled()
                best_fitness = self._evaluator.evaluate(best_solution)
                self._logger.info('Iterations: %d, Best fitness: %s, Solution: %s' % (eval, best_fitness, best_solution))
                print('Iterations: %d, Best fitness: %s, Solution: %s' % (eval, best_fitness, best_solution))

            progress_bar.set_postfix_str(f"Fitness: {fitness}, Solution: {point}")
            progress_bar.update(eval+1)
            self._save_session(eval, best_fitness, best_solution)
        # Once we have evaluated all the points, get the best one we have seen so far

        print('Retrieving best solution...')
        self._save_session(n_evaluations, best_fitness, best_solution)
        self._logger.info('Best solution: %s' % (best_solution,))
        return best_solution, best_fitness

    def _get_data(self):
        data = {
            'k_explore': self._k_explore,
            'n_samples': self._n_samples,
            'eval_neighbours': self._eval_neighbours,
        }
        return data

    def _load_data(self, data):
        self._k_explore = data['k_explore']
        self._n_samples = data['n_samples']
        self._eval_neighbours = data['eval_neighbours']

    def _save_session(self, eval, best_fitness, best_solution):
        session_filename = f'ntbea_session_{self._world_model}_{self._agent_type}'
        session_filepath = join(self.session_dir, f'{session_filename}')

        save_data = {
            'world_model': self._world_model,
            'agent_type': self._agent_type,
            'iterations': eval,
            'best_fitness': best_fitness,
            'best_solution': best_solution,
            'ntbea_data': self._get_data(),
            'tuple_landscape_data': self._tuple_landscape.get_data(),
            'search_space_data': self._search_space.get_data(),
            'mutate_data': self._mutator.get_data()
        }
        with open(f'{session_filepath}.pickle', 'wb') as file:
            pickle.dump(save_data, file, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_session(self):
        session_filename = self.config['ntbea_tuning']['session_name']
        session_filepath = join(self.session_dir, f'{session_filename}.pickle')

        if exists(session_filepath):
            with open(f'{session_filepath}', 'rb') as file:
                save_data = pickle.load(file)

                if self._world_model != save_data['world_model'] and self._agent_type != save_data['agent_type']:
                    raise Exception(f'Inconsistency between loaded world model/agent {save_data["world_model"]}/{save_data["agent_type"]} and actual in config {self._world_model}/{self._agent_type}')

                self._world_model = save_data['world_model']
                self._agent_type = save_data['agent_type']
                self._load_data(save_data['ntbea_data'])
                self._search_space.load_data(save_data['search_space_data'])
                self._mutator.load_data(save_data['mutate_data'])
                self._tuple_landscape.load_data(save_data['tuple_landscape_data'])

                return save_data['iterations'], save_data['best_solution'], save_data['best_fitness']
        raise Exception(f'NO SESSION FOUND WITH {session_filepath}')
