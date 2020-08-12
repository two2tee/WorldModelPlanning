import gym
import random
from environment.base_environment import BaseEnvironment


class CarRacingEnvironment(BaseEnvironment):
    def __init__(self, config):
        super().__init__(config)
        self.seed = None
        self.is_skip_zoom = False

    def reset(self, seed=None):
        if self.environment is None:
            self.environment = gym.make(self.game_name)

        self.seed = seed if seed else random.randint(0, 2 ** 31 - 1)
        self.environment.seed(self.seed)
        state = self.environment.reset()
        return self._skip_zoom() if self.is_skip_zoom else state

    def _skip_zoom(self):
        state = None
        for _ in range(50):  # Skip zoom
            state, _, _, _ = self.step([0, 0, 0])
        return state
