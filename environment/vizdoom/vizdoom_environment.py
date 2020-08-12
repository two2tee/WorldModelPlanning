from environment.vizdoom.vizdoomgym.vizdoomgym.envs.vizdoom_env_definitions import VizdoomTakeCover
from environment.base_environment import BaseEnvironment

class VizdoomEnvironment(BaseEnvironment):
    def __init__(self, config):
        super().__init__(config)
        self.environment = VizdoomTakeCover()

    def reset(self, seed=None):\
        return self.environment.reset()

    def step(self, action):
        obs, reward, done, info = self.environment.step(self._convert_action(action))
        reward = -1 if done else reward
        return obs, reward, done, info

    def _convert_action(self, action):
        return action[0] if action[0] != -1 else None

