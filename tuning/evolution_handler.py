""" Evolution Handler is used to streamline evolution operators across RHEA and RMHC while enabling NTBEA tuning """
#  Copyright (c) 2020, - All Rights Reserved
#  This file is part of the Evolutionary Planning on a Learned World Model thesis.
#  Unauthorized copying of this file, via any medium is strictly prohibited without the consensus of the authors.
#  Written by Thor V.A.N. Olesen <thorolesen@gmail.com> & Dennis T.T. Nguyen <dennisnguyen3000@yahoo.dk>.

import json
import random
import numpy as np
from planning.interfaces.individual import Individual


class EvolutionHandler:
    def __init__(self, horizon):

        with open('config.json') as config_file:
            config = json.load(config_file)

        seed = None if config['evolution_handler']['seed'] == "None" else config['evolution_handler']['seed']
        random.seed(seed)  # Ensure deterministic random generator

        self.mutation_probability = config['evolution_handler']['mutation_probability'] if "mutation_probability" in config['evolution_handler'] else 1 / self.horizon
        self.tournament_percentage = config['evolution_handler']['tournament_percentage']

        self.mutation_methods = {'single_uniform': self._single_uniform_mutation,
                                 'all_uniform': self._all_uniform_mutation,
                                 'subset_mutation': self._subset_mutation}

        self.crossover_methods = {'uniform': self._uniform_crossover,
                                  '1_bit': self._one_bit_crossover,
                                  '2_bit': self._two_bit_crossover}

        self.selection_methods = {'uniform': self._uniform_selection,
                                  'tournament': self._tournament_selection,
                                  'rank': self._rank_selection,
                                  'roulette': self._roulette_selection}

        self.genetic_operator = {'crossover': 'cross',
                                 "mutation": 'mut',
                                 "crossover_mutation": 'crossmut'}

        self.horizon = horizon

        self.default_mutation_method = config['evolution_handler']['mutation_method']
        self.default_crossover_method = config['evolution_handler']['crossover_method']
        self.default_selection_method = config['evolution_handler']['selection_method']
        self.default_genetic_operator = config['evolution_handler']['genetic_operator']

    def get_genetic_operator(self, operator=None):
        operator = self.default_genetic_operator if operator is None else operator
        return self.genetic_operator[operator]

    def get_mutation_operator(self, mutation_method=None):
        mutation_method = self.default_mutation_method if mutation_method is None else mutation_method
        return self.mutation_methods[mutation_method]

    def get_crossover_operator(self, crossover_method=None):
        crossover_method = self.default_crossover_method if crossover_method is None else crossover_method
        return self.crossover_methods[crossover_method]

    def get_selection_type(self, selection_method=None):
        selection_method = self.default_selection_method if selection_method is None else selection_method
        return self.selection_methods[selection_method]

    # SELECTION METHODS
    def _uniform_selection(self, population):
        return np.random.choice(population, size=2, replace=False)

    def _tournament_selection(self, population):
        tournament_population = random.sample(population, k=int(len(population) * self.tournament_percentage)) \
                                if len(population) > 2 else population  # Use all if population is too small
        population_sorted_by_fitness = sorted(tournament_population, key=lambda individual: individual.fitness, reverse=True)
        return population_sorted_by_fitness[0], population_sorted_by_fitness[1]

    #  https://stackoverflow.com/questions/20290831/how-to-perform-rank-based-selection-in-a-genetic-algorithm
    def _rank_selection(self, population):
        population_sorted_by_fitness = sorted(population, key=lambda individual: individual.fitness)  # ascending
        n = len(population)
        rank_sum = (n / 2) * (n + 1)  # Gauss formula sums of integers 1 to n

        # Make rank wheel
        rank_wheel = list([i / rank_sum for i in range(1, len(population) + 1)])  # 1/rank_sum + 2/rank_sum + ... + n/rank_sum = 1

        # Select parents based on rank wheel
        parent_a = population_sorted_by_fitness[np.random.choice(len(population), p=rank_wheel)]
        parent_b = None
        while parent_b is None or parent_b == parent_a:
            parent_b = population_sorted_by_fitness[np.random.choice(len(population), p=rank_wheel)]
        return parent_a, parent_b

    def _roulette_selection(self, population):
        population_sorted_by_fitness = sorted(population, key=lambda individual: individual.fitness)  # ascending

        fitnesses = [i.fitness for i in population_sorted_by_fitness]

        if any(fitness < 0 for fitness in fitnesses):  # Normalize negative values to postive by summing smallest value
            smallest_fitness = np.amin(fitnesses)
            fitnesses = list(map(lambda fitness: fitness+abs(smallest_fitness)+0.01, fitnesses))

        fitness_sum = sum(fitnesses)

        # Make roulette wheel
        roulette_wheel = list([fitness / fitness_sum for fitness in fitnesses])

        # Select parents based on rank wheel
        parent_a = population_sorted_by_fitness[np.random.choice(len(population), p=roulette_wheel)]
        parent_b = None
        while parent_b is None or parent_b == parent_a:
            parent_b = population_sorted_by_fitness[np.random.choice(len(population), p=roulette_wheel)]
        return parent_a, parent_b

    # CROSSOVER METHODS
    def _uniform_crossover(self, parent_a, parent_b, generation):
        action_sequence = [parent_a.action_sequence[i] if random.uniform(0, 1) < 0.5 else  # Bernoulli uniform cross over
                           parent_b.action_sequence[i] for i in range(self.horizon)]
        return Individual(action_sequence, age=generation)

    def _one_bit_crossover(self, parent_a, parent_b, generation):
        return self._n_bit_crossover(parent_a, parent_b, generation, bit=1)

    def _two_bit_crossover(self, parent_a, parent_b, generation):
        return self._n_bit_crossover(parent_a, parent_b, generation, bit=2)

    def _n_bit_crossover(self, parent_a, parent_b, generation, bit):
        horizon_length = len(parent_a.action_sequence)
        choices = np.arange(1, horizon_length)
        bit = horizon_length - 1 if bit >= horizon_length else bit
        split_points = np.sort(np.random.choice(choices, size=bit, replace=False))
        action_sequence = []
        start_index = 0
        parents = np.random.choice([parent_a, parent_b], size=2, replace=False)  #
        i = 0
        for split_point in split_points:
            parent = parents[i % 2]  # alternate switching
            action_sequence.extend(parent.action_sequence[start_index:split_point])
            start_index = split_point
            i += 1

        if len(action_sequence) != horizon_length:
            action_sequence.extend(parents[i % 2].action_sequence[start_index: horizon_length])

        return Individual(action_sequence, age=generation)

    # MUTATION METHODS
    def _single_uniform_mutation(self, environment, child):
        # Mutate single action
        random_index = random.randrange(self.horizon)
        child.action_sequence[random_index] = environment.sample() if random.uniform(0,1) < self.mutation_probability else \
                                              child.action_sequence[random_index]

    def _all_uniform_mutation(self, environment, child):
        # Mutate all actions uniformly randomly
        child.action_sequence = [environment.sample() if random.uniform(0, 1) < self.mutation_probability else
                                 action for action in child.action_sequence]

    def _subset_mutation(self, environment, child):
        # Mutate random number of actions
        mutation_indices = {random.randrange(self.horizon)}
        other_mutation_indices = set([i for i, _ in enumerate(child.action_sequence)
                                     if random.uniform(0, 1) < self.mutation_probability])
        mutation_indices = mutation_indices.union(other_mutation_indices)
        mutations = [environment.sample() for _ in range(len(mutation_indices))]
        for i in mutation_indices:
            child.action_sequence[i] = mutations.pop(0)

