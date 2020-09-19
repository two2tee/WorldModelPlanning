import gym
import environment.vizdoom.vizdoom_implementation.vizdoomgym  # DO NOT REMOVE - Needed to register custom gym envs
from environment.base_environment import BaseEnvironment


class VizdoomEnvironment(BaseEnvironment):
    def __init__(self, config):
        super().__init__(config)
        self.environment = gym.make('VizdoomTakeCover-v0', health=True)

    def reset(self, seed=None):
        super(VizdoomEnvironment, self).reset()
        if seed:
            self.environment.seed(seed)
        states = self.environment.reset()
        # self.health = states[1]
        return states[0]  # Frame

    def step(self, action, ignore_is_done=False):
        states, reward, done, info = self.environment.step(self.action_sampler.convert_action(action[0]))
        frame = states[0]
        return frame, reward, done, info
