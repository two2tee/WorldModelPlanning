import numpy as np


class IterationResult:
    def __init__(self, iteration=0):
        self.iteration = iteration
        self.test_name = None
        self.seed = None
        self.total_trials = 0
        self.trials_max_rewards = []
        self.trials_rewards = []
        self.mdrnn_test_losses = {}

    def get_average_total_reward(self):
        return np.mean(self.trials_rewards) if len(self.trials_rewards) > 0 else 0

    def get_average_max_reward(self):
        return np.mean(self.trials_max_rewards) if len(self.trials_max_rewards) > 0 else 0

    def to_dict(self):
        return dict((name, getattr(self, name)) for name in dir(self) if not name.startswith('__') and not callable(getattr(self, name)))

    @staticmethod
    def to_obj(data_dict):
        iteration_result = IterationResult()
        for key, value in data_dict.items():
            setattr(iteration_result, key, value)
        return iteration_result

    def __str__(self) -> str:
        return f'\n\nIteration {self.iteration} | {self.mdrnn_test_losses} | total trials in test: {self.total_trials} ' \
               f'| test name {self.test_name}' \
               f'\navg max reward: {self.get_average_max_reward()} | avg reward: {self.get_average_total_reward()}\n\n'
