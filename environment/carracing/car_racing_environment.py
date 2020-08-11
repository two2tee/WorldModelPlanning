import gym
import random
from environment.base_environment import BaseEnvironment

class CarRacingEnvironment(BaseEnvironment):
    def __init__(self, config):
        super().__init__(config)
        self.seed = None

    def reset(self, seed=None):
        if self.environment is None:
            self.environment = gym.make(self.game_name)

        self.seed = seed if seed else random.randint(0, 2 ** 31 - 1)
        self.environment.seed(self.seed)
        return self.environment.reset()
