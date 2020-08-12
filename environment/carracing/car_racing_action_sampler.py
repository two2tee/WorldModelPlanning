import random

import numpy as np

from environment.actions.base_action_sampler import BaseActionSampler, brownian_sample


class CarRacingActionSampler(BaseActionSampler):
    def __init__(self, config):
        super().__init__(config, num_actions=3)
        self.steer_delta = self.config['simulated_environment']['car_racing']['steer_delta']
        self.gas_delta = self.config['simulated_environment']['car_racing']['gas_delta']
        self.max_gas = self.config['simulated_environment']['car_racing']['max_gas']
        self.max_brake = self.config['simulated_environment']['car_racing']['max_brake']

    def sample(self, previous_action=None):  # Sampling: [ steer, gas, brake ] = [ [-1, +1] , [0, 1], [0, 1] ]
        return self._continous_sample(previous_action) if not self.is_discretize_sampling else self.discrete_sample()

    def _continous_sample(self, previous_action=None):
        steer = np.random.uniform(low=-1, high=1) if previous_action is None else \
                np.random.choice([max(previous_action[0] - self.steer_delta, -1),
                                  previous_action[0],
                                  min(previous_action[0] + self.steer_delta, 1)])

        gas = np.random.uniform(low=self.max_brake, high=self.max_gas) if previous_action is None else \
              np.random.choice([max(previous_action[1] - self.gas_delta, -1),
                                previous_action[1],
                                min(previous_action[1] + self.gas_delta, 1)])

        # Brake: negative sign of gas to avoid simultaneous brake/gas driving
        gas = gas if gas > 0 else 0
        brake = abs(gas) if gas < 0 else 0
        return [steer, gas, brake]

    def discrete_sample(self):
        steer_steps = [round(e, 1) for e in np.arange(start=-1.0, stop=1.0, step=0.1)]
        gas_steps = [round(e, 1) for e in np.arange(start=-1.0, stop=1.0, step=0.2)]
        steer, gas = np.random.choice(steer_steps), np.random.choice(gas_steps)
        gas = gas if gas > 0 else 0
        brake = abs(gas) if gas < 0 else 0
        return [steer, gas, brake]

    def brownian_sample(self, previous_action):  # a_{t+1} = a_t + sqrt(dt) N(0, 1)
        new_action = [0, 0, 0]
        new_action[0] = brownian_sample(previous_action[0], upper=-1, lower=1)
        new_action[1] = brownian_sample(previous_action[1], upper=0, lower=1)
        new_action[2] = brownian_sample(previous_action[1], upper=0, lower=1)
        return new_action

    def discrete_delta_sample(self, previous_action=None):
        actions = self.discrete_action_space(previous_action)
        random_index = random.randrange(len(actions))
        return actions[random_index]

    def discrete_action_space(self, action=None):
        actions = set()
        steer_steps = np.arange(start=-1.0, stop=1.0, step=self.steer_delta) if action is None else [max(action[0] - self.steer_delta, -1), action[0], min(action[0] + self.steer_delta, 1)]
        gas_steps = np.arange(start=-1.0, stop=1.0, step=self.gas_delta) if action is None else [max(action[1] - self.gas_delta, -1), action[1], min(action[1] + self.gas_delta, 1)]
        steer_steps, gas_steps = [round(e, 1) for e in steer_steps], [round(e, 1) for e in gas_steps]  # Remove decimal precision

        for steer in steer_steps:
            for gas in gas_steps:
                actions.add((steer, gas, 0)) if gas > 0 else actions.add((steer, 0, abs(gas)))  # negative sign gas = brake

        return [list(a) for a in actions]