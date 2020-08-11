""" Environment interface with custom sampling and replication """
#  Copyright (c) 2020, - All Rights Reserved
#  This file is part of the Evolutionary Planning on a Learned World Model thesis.
#  Unauthorized copying of this file, via any medium is strictly prohibited without the consensus of the authors.
#  Written by Thor V.A.N. Olesen <thorolesen@gmail.com> & Dennis T.T. Nguyen <dennisnguyen3000@yahoo.dk>.

import gym
import random
import environment.actions.action_sampler_factory as action_sampler


class EnvironmentWrapper:
    def __init__(self, config, seed=None):
        self.seed = seed
        self.config = config
        self.environment = None
        self.game_name = config['game']
        self.action_sampler = action_sampler.get_action_sampler(config)

    def step(self, action):
        if self.environment is None:
            raise Exception('Cannot call step before reset.')
        return self.environment.step(action)

    def reset(self, seed=None):
        if self.environment is None:
            self.environment = gym.make(self.game_name)

        self.seed = seed if seed else random.randint(0, 2 ** 31 - 1)
        self.environment.seed(self.seed)
        return self.environment.reset()

    def render(self):
        self.environment.render()

    def sample(self):
        return self.environment.sample()

    def close(self):
        self.environment.close()
