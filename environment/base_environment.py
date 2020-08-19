""" Environment interface with custom sampling and replication """
#  Copyright (c) 2020, - All Rights Reserved
#  This file is part of the Evolutionary Planning on a Learned World Model thesis.
#  Unauthorized copying of this file, via any medium is strictly prohibited without the consensus of the authors.
#  Written by Thor V.A.N. Olesen <thorolesen@gmail.com> & Dennis T.T. Nguyen <dennisnguyen3000@yahoo.dk>.

import environment.actions.action_sampler_factory as action_sampler


class BaseEnvironment:
    def __init__(self, config):
        self.config = config
        self.environment = None
        self.game_name = config['game']
        self.action_sampler = action_sampler.get_action_sampler(config)
        self._is_done = False

    def step(self, action):
        if self.environment is None:
            raise Exception('Cannot call step before reset.')
        if self._is_done:
            raise Exception('Cannot step since game is done. Please call reset.')

        state, reward, is_done, info = self.environment.step(action)
        self._is_done = is_done
        return state, reward, is_done, info

    def reset(self, seed=None):
        self._is_done = False

    def render(self):
        if self._is_done:
            raise Exception('Cannot render since game is done. Please call reset.')
        self.environment.render()

    def sample(self):
        return self.action_sampler.sample()

    def close(self):
        self.environment.close()
