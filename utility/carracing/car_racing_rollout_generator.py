import os
import random
import gym
import numpy as np

from tqdm import tqdm
from gym.envs.box2d.car_dynamics import Car
from utility.base_rollout_generator import BaseRolloutGenerator


class RolloutGenerator(BaseRolloutGenerator):
    def __init__(self, config, data_output_dir):
        super().__init__(config,  data_output_dir)

    def _standard_rollout(self, environment, thread, current_rollout, rollouts):
        action = [0, 0, 0]
        model = self._get_model() if self.config["data_generator"]['car_racing']["is_ha_agent_driver"] else None
        obs, _ = self._reset(environment)
        actions_rollout, states_rollout, reward_rollout, is_done_rollout = [], [], [], []

        progress_description = f"Data generation for {self.config['game']} | thread: {thread} | rollout: {current_rollout}/{rollouts}"
        for _ in tqdm(range(self.sequence_length + 1), desc=progress_description, position=thread - 1):
            obs, reward, done, info, action = self._step(environment, obs, action, model)
            obs = self._compress_frame(obs, is_resize=True)
            environment.environment.viewer.window.dispatch_events()
            actions_rollout.append(action)
            states_rollout.append(obs)
            reward_rollout.append(reward)
            is_done_rollout.append(done)
            if done:
                break
        environment.close()
        return actions_rollout, states_rollout, reward_rollout, is_done_rollout

    def _reset(self, environment):
        # Generate random tracks and initial car positions
        environment.reset()

        # Worse total reward by randomizing car position
        car_position = np.random.randint(len(environment.track))
        environment.environment.car = Car(environment.environment.world, *environment.environment.track[car_position][1:4])

        # Garbage collection of events in viewer
        obs, _, _, _ = environment.step([0,0,0])
        environment.environment.viewer.window.dispatch_events()
        return obs, car_position

    def _step(self, environment, obs, previous_action, model=None):
        if model:
            z, mu, logvar = model.encode_obs(obs)
            action = model.get_action(z)
        else:
            action = self.action_sampler.brownian_sample(previous_action)
        obs, reward, done, info = environment.step(action)

        return obs, reward, done, info, action

    def _get_model(self):
        from utility.carracing.ha_implementation.model import Model
        model = Model(load_model=True)
        model.load_model(f"{os.getcwd()}/utility/carracing/ha_implementation/log/carracing.cma.16.64.best.json")
        return model
