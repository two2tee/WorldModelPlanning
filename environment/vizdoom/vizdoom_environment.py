from environment.vizdoom.vizdoomgym.vizdoomgym.envs.vizdoom_env_definitions import VizdoomTakeCover
from environment.base_environment import BaseEnvironment


class VizdoomEnvironment(BaseEnvironment):
    def __init__(self, config):
        super().__init__(config)
        self.environment = VizdoomTakeCover(health=True)
        self.health = None


    def reset(self, seed=None):
        if seed:
            self.environment.seed(seed)
        states = self.environment.reset()
        self.health = states[1]
        return states[0]  # Frame

    def step(self, action):
        states, reward, done, info = self.environment.step(self._convert_action(action))
        frame = states[0]
        new_health = states[1]

        reward = -1.0 if done or self._is_health_drop(new_health) else reward
        self.health = new_health if self._is_health_drop(new_health) else self.health

        return frame, reward, done, info

    def _convert_action(self, action):
        return action[0] if action[0] != -1 else None

    def _is_health_drop(self, new_health):
        return new_health != self.health
