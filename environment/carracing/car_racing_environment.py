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
        self.is_standard_reward = self.config['real_environment']['car_racing']['standardize_reward']

    def step(self, action, ignore_is_done=False):
        state, reward, is_done, info = super().step(action, ignore_is_done)
        reward = self._standardize_reward(reward) if self.is_standard_reward else reward
        return state, self._standardize_reward(reward), is_done, info

    def reset(self, seed=None):
        super(CarRacingEnvironment, self).reset()
        if self.environment is None:
            self.environment = gym.make(self.game_name)

        self.seed = seed if seed else random.randint(0, 2 ** 31 - 1)
        self.environment.seed(self.seed)
        state = self.environment.reset()
        state = self._randomize_car_pos() if self.is_random_inital_car_position else state
        state = self._skip_zoom() if self.is_skip_zoom else state
        return state

    def _skip_zoom(self):
        return [self.environment.step([0, 0, 0])[0] for _ in range(50)][-1]

    def _randomize_car_pos(self):
        random_car_position = np.random.randint(len(self.environment.env.track))
        self.environment.car = Car(self.environment.world, *self.environment.track[random_car_position][1:4])
        obs, _, _, _ = self.step([0, 0, 0])
        return obs

    def _standardize_reward(self, reward):
        reward = 3.0 if reward > 3 else reward
        return round(reward, 1)