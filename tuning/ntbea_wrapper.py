""" Interface for NTBEA tuning with world model implementation """
#  Copyright (c) 2020, - All Rights Reserved
#  This file is part of the Evolutionary Planning on a Learned World Model thesis.
#  Unauthorized copying of this file, via any medium is strictly prohibited without the consensus of the authors.
#  Written by Thor V.A.N. Olesen <thorolesen@gmail.com> & Dennis T.T. Nguyen <dennisnguyen3000@yahoo.dk>.

import numpy as np
from abc import ABC
from tuning.ntbea.common import DefaultMutator
from tuning.ntbea.ntbea import SearchSpace, Evaluator, NTupleLandscape, NTupleEvolutionaryAlgorithm


class PlanningSearchSpace(SearchSpace, ABC):
    def __init__(self, parameters):
        self._search_space = parameters
        self._max_possible_options = self._get_max_possible_options(parameters)
        self._number_of_parameter_types = len(self._search_space)
        super(PlanningSearchSpace, self).__init__("Planning Params Search Space", ndims=self._number_of_parameter_types)


    def get_random_point(self):
        point = []
        for param_dimension in self._search_space:
            parameter_value = np.random.choice(param_dimension)  # np.random.randint(0, len(param_dimension))
            point.append(parameter_value)
        return np.array(point)

    def get_valid_values_in_dim(self, dim):
        return self._search_space[dim]

    def get_size(self):
        return self._number_of_parameter_types ** self._max_possible_options

    def _get_max_possible_options(self, parameter_space):
        max_options = 0
        for params in parameter_space:
            max_options = len(params) if len(params) > max_options else max_options
        return max_options

    def get_data(self):
        data = {'search_space': self._search_space}
        return data

    def load_data(self, data):
        print('Loaded search space')
        self._search_space = data['search_space']
        self._max_possible_options = self._get_max_possible_options(self._search_space)
        self._ndims = self._max_possible_options
        self._number_of_parameter_types = len(self._search_space)


class PlanningEvaluator(Evaluator):
    def __init__(self, planning_test_suite, agent_type):
        super(PlanningEvaluator, self).__init__("Planning Evalutator")
        self.test_suite = planning_test_suite
        self.agent_type = agent_type

    def evaluate(self, point):
        self._set_agent_params(point)
        self.test_suite.trials = 1
        self.test_suite.is_ntbea_tuning = True

        run_all = True
        if run_all:
            reward = self.test_suite.run_tests()
        else:
            _, _, trial_rewards, _, _ = self.test_suite.run_specific_test(test_name='right_turn_planning_test') #planning_whole_random_track
            reward = trial_rewards[0]

        return reward

    def _set_agent_params(self, point):
        evolution_handler = self.test_suite.planning_agent.evolution_handler

        self.test_suite.planning_agent.latent = None
        self.test_suite.planning_agent.hidden = None
        self.test_suite.planning_agent.current_elite = None
        self.test_suite.planning_agent.elite_history = None
        self.test_suite.planning_agent.is_shift_buffer = point[0] == 'True'  # Due to string conversion...
        self.test_suite.planning_agent.horizon = int(point[1])
        self.test_suite.planning_agent.evolution_handler.horizon = int(point[1])
        self.test_suite.planning_agent.max_generations = int(point[2])
        self.test_suite.planning_agent.is_rollout = point[3] == 'True'
        self.test_suite.planning_agent.max_rollouts = int(point[4])
        self.test_suite.planning_agent.mutation_operator = evolution_handler.get_mutation_operator(point[5])

        if self.agent_type == "RHEA":
            self.test_suite.planning_agent.population = None
            self.test_suite.planning_agent.population_size = int(point[6])
            self.test_suite.planning_agent.genetic_operator = evolution_handler.get_genetic_operator(point[7])
            self.test_suite.planning_agent.selection_type = evolution_handler.get_selection_type(point[8])
            self.test_suite.planning_agent.crossover_operator = evolution_handler.get_crossover_operator(point[9])


class PlanningNTBEAWrapper:
    def __init__(self, config, planning_tester):
        self.config = config
        self.planning_tester = planning_tester
        self.mutation_point_probability = self.config['ntbea_tuning']['mutation_point_probability']
        self.ucb_random_noise = self.config['ntbea_tuning']['ucb_random_noise']
        self.explore_rate = self.config['ntbea_tuning']['explore_rate']
        self.eval_neighbours = self.config['ntbea_tuning']['eval_neighbours']
        self.samples_per_eval = self.config['ntbea_tuning']['samples_per_eval']
        self.iterations = self.config['ntbea_tuning']['iterations']
        self.agent_type = self.config['planning']['planning_agent']
        self.world_model = self.config['experiment_name']

    def run_ntbea(self):
        print(f'Running NTBEA for planning agent: {self.agent_type} with world model: {self.world_model}'
              f'\nIterations: {self.iterations} | explore_rate: {self.explore_rate} | mutation_probability: {self.mutation_point_probability}')

        # Search Space and evaluator
        parameter_space = self._get_parameters(self.agent_type)
        search_space = PlanningSearchSpace(parameter_space)
        planning_evaluator = PlanningEvaluator(self.planning_tester, self.agent_type)

        # 1-tuple, 2-tuple and N-tuple
        tuple_config = [1, 2, search_space._number_of_parameter_types]
        tuple_landscape = NTupleLandscape(search_space, tuple_config, self.ucb_random_noise)

        # point mutator (param config mutation)
        mutator = DefaultMutator(search_space, mutation_point_probability=self.mutation_point_probability)

        # NTBEA Algorithm
        evolutionary_algorithm = NTupleEvolutionaryAlgorithm(tuple_landscape, planning_evaluator, search_space, mutator,
                                                             self.explore_rate, self.samples_per_eval,self.eval_neighbours,
                                                             world_model=self.world_model, agent_type=self.agent_type,
                                                             config=self.config)

        best_params_config, fitness = evolutionary_algorithm.run(self.iterations)

        self.print_results(best_params_config, fitness)

    def _get_parameters(self, agent_type):
        shift_buffer_options = self.config['ntbea_tuning']['shift_buffer_options']
        horizon_options = self.config['ntbea_tuning']['horizon_options']
        generation_options = self.config['ntbea_tuning']['generation_options']
        is_rollout_options = self.config['ntbea_tuning']["is_rollout_options"]
        max_rollout_options = self.config['ntbea_tuning']['max_rollout_options']
        mutation_options = self.config['ntbea_tuning']['mutation_options']
        population_size_options = self.config['ntbea_tuning']['RHEA_population_size']
        genetic_operator_options = self.config['ntbea_tuning']['RHEA_genetic_operator_options']
        selection_options = self.config['ntbea_tuning']['RHEA_selection_options']
        crossover_methods_options = self.config['ntbea_tuning']['RHEA_crossover_methods_options']

        # ##DO NOT CHANGE ORDER IN ARRAY
        search_space = [shift_buffer_options, horizon_options, generation_options, is_rollout_options, max_rollout_options, mutation_options]

        if agent_type == "RHEA":
            search_space.append(population_size_options)
            search_space.append(genetic_operator_options)
            search_space.append(selection_options)
            search_space.append(crossover_methods_options)
        # ##
        print(f'search space: {search_space}')
        return search_space

    def print_results(self, best_params_config, fitness):
        print(f'COMPLETED NTBEA for planning agent: {self.agent_type} with world model: {self.world_model}')
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
                      if self.agent_type == "RHEA" else stats
        print(stats)

