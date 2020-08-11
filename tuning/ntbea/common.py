""" NTBEA Dependencies based on https://github.com/bam4d/NTBEA """

#  Copyright (c) 2020, - All Rights Reserved
#  This file is part of the Evolutionary Planning on a Learned World Model thesis.
#  Unauthorized copying of this file, via any medium is strictly prohibited without the consensus of the authors.
#  Written by Thor V.A.N. Olesen <thorolesen@gmail.com> & Dennis T.T. Nguyen <dennisnguyen3000@yahoo.dk>.

import copy
import random
import numpy as np

from tuning.ntbea.ntbea import Mutator


class DefaultMutator(Mutator):
    """
    A simple mutator used by Examples
    """

    def __init__(self, search_space, swap_mutate=False, random_chaos_mutate=False, mutation_point_probability=1.0, flip_at_least_one=True):
        super(DefaultMutator, self).__init__("Default Mutator")

        self._search_space = search_space
        self._swap_mutate = swap_mutate
        self._random_chaos_mutate = random_chaos_mutate
        self._mutation_point_probability = mutation_point_probability
        self._flip_at_least_one = flip_at_least_one

    def _swap_mutation(self, point):
        length = len(point)

        idx = random.sample(range(length), size=2)

        a = point[idx[0]]
        b = point[idx[1]]

        point[idx[0]] = b
        point[idx[1]] = a

        return point

    def _mutate_value(self, point, dim):
        """
        mutate the value of x at the given dimension 'dim'
        """
        point = list(point)
        params = self._search_space.get_valid_values_in_dim(dim)
        choice = np.random.choice(params)
        point[dim] = choice
        return np.array(point)

    def mutate(self, point):

        new_point = copy.deepcopy(point)
        length = len(point)

        # Perform swap mutation operation
        if self._swap_mutate:
            return self._swap_mutation(point)

        # Random mutation i.e just return a random search point
        if self._random_chaos_mutate:
            return self._search_space.get_random_point()

        # For each of the dimensions, we mutate it based on mutation_probability
        for dim in range(length):
            if self._mutation_point_probability > np.random.uniform():
                new_point = self._mutate_value(new_point, dim)


        # If we want to force flip at least one of the points then we do this here
        if self._flip_at_least_one:
            new_point = self._mutate_value(new_point, random.sample(range(length), 1)[0])
        return new_point

    def get_data(self):
        data = {'search_space': self._search_space,
                'swap_mutate': self._swap_mutate,
                'random_chaos_mutate': self._random_chaos_mutate,
                'mutation_probability': self._mutation_point_probability,
                'flip_at_least_once': self._flip_at_least_one
                }
        return data

    def load_data(self, data):
        print('Loaded mutate data')
        self._search_space = data['search_space']
        self._swap_mutate = data['swap_mutate']
        self._random_chaos_mutate = data['random_chaos_mutate']
        self._mutation_point_probability = data['mutation_probability']
        self._flip_at_least_one = data['flip_at_least_once']







