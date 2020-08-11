""" Interface for Random Mutation Hill Climbing for non-simulated environment"""
#  Copyright (c) 2020, - All Rights Reserved
#  This file is part of the Evolutionary Planning on a Learned World Model thesis.
#  Unauthorized copying of this file, via any medium is strictly prohibited without the consensus of the authors.
#  Written by Thor V.A.N. Olesen <thorolesen@gmail.com> & Dennis T.T. Nguyen <dennisnguyen3000@yahoo.dk>.

from abc import ABC, abstractmethod


class AbstractRandomMutationHillClimbing(ABC):
    action = int

    @abstractmethod
    def __init__(self, horizon, max_generations, is_shift_buffer, is_rollout, max_rollouts=None, mutation_probability=None) -> None:
        self.horizon = horizon
        self.is_rollout = is_rollout
        self.num_rollouts = max_rollouts
        self.is_shift_buffer = is_shift_buffer
        self.max_generations = max_generations
        self.rollout_length = int(self.horizon / 2)
        self.mutation_probability = 1 / self.horizon if mutation_probability is None else mutation_probability
        super().__init__()

    @abstractmethod
    def search(self, environment) -> action:
        pass
