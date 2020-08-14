import gym
import random
import numpy as np
from gym.envs.box2d.car_dynamics import Car
from environment.base_environment import BaseEnvironment


class CarRacingEnvironment(BaseEnvironment):
    def __init__(self, config):
        super().__init__(config)
        self.seed = None
        self.is_skip_zoom = self.config['real_environment']['car_racing']['skip_zoom']
        self.is_random_inital_car_position = self.config['real_environment']['car_racing']['random_intial_car_pos']

    def reset(self, seed=None):
        if self.environment is None:
            self.environment = gym.make(self.game_name)

        self.seed = seed if seed else random.randint(0, 2 ** 31 - 1)
        self.environment.seed(self.seed)
        state = self.environment.reset()
        state = self._randomize_car_pos if self.is_random_inital_car_position else state
        state = self._skip_zoom() if self.is_skip_zoom else state
        return state

    def _skip_zoom(self):
        return [self.environment.step([0, 0, 0])[0] for _ in range(50)][-1]

    def _randomize_car_pos(self):
        random_car_position = np.random.randint(len(self.environment.env.track))
        self.environment.car = Car(self.environment.world, *self.environment.track[random_car_position][1:4])
        obs, _, _, _ = self.step([0, 0, 0])
        return obs

