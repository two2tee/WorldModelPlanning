""" Interface for Random Mutation Hill Climbing for simulated environment """
#  Copyright (c) 2020, - All Rights Reserved
#  This file is part of the Evolutionary Planning on a Learned World Model thesis.
#  Unauthorized copying of this file, via any medium is strictly prohibited without the consensus of the authors.
#  Written by Thor V.A.N. Olesen <thorolesen@gmail.com> & Dennis T.T. Nguyen <dennisnguyen3000@yahoo.dk>.

from abc import ABC, abstractmethod


class AbstractGradientHillClimbing(ABC):
    action = int

    @abstractmethod
    def __init__(self, horizon, max_steps, is_shift_buffer, learning_rate) -> None:
        self.horizon = horizon
        self.is_shift_buffer = is_shift_buffer
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        super().__init__()

    @abstractmethod
    def search(self, simulated_environment, initial_latent_state, initial_hidden_state) -> action:
        pass
