""" Interface for Monte Carlo Tree Search for simulated environment """
#  Copyright (c) 2020, - All Rights Reserved
#  This file is part of the Evolutionary Planning on a Learned World Model thesis.
#  Unauthorized copying of this file, via any medium is strictly prohibited without the consensus of the authors.
#  Written by Thor V.A.N. Olesen <thorolesen@gmail.com> & Dennis T.T. Nguyen <dennisnguyen3000@yahoo.dk>.

from abc import ABC, abstractmethod


class AbstractTreeSearch(ABC):
    action = int

    @abstractmethod
    def __init__(self, temperature, max_rollouts, rollout_length=None) -> None:
        self.temperature = temperature
        self.max_rollouts = max_rollouts
        self.rollout_length = rollout_length
        self.root = None
        super().__init__()

    @abstractmethod
    def search(self, simulated_environment, initial_latent_state, initial_hidden_state) -> action:
        pass
