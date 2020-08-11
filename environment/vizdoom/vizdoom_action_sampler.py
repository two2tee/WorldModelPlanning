from environment.actions.base_action_sampler import BaseActionSampler


class VizdoomActionSampler(BaseActionSampler):
    def __init__(self, config):
        super().__init__(config, num_actions=-1)

    def sample(self, previous_action=None):
        return NotImplemented

    def brownian_sample(self, previous_action):
        return NotImplemented

    def discrete_delta_sample(self, previous_action=None):
        return NotImplemented

    def discrete_action_space(self, action=None):
        return NotImplemented
