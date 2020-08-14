from environment.actions.base_action_sampler import BaseActionSampler
import random


class VizdoomActionSampler(BaseActionSampler):
    def __init__(self, config):
        super().__init__(config, num_actions=1)

    def sample(self, previous_action=None):
        return [random.choice([1, 2])]  # set -1 as lowerbound if no operation action is included

    def brownian_sample(self, previous_action):
        return self.sample()

    def discrete_delta_sample(self, previous_action=None):
        return self.sample()

    def discrete_action_space(self, action=None):
        return [1, 2]
