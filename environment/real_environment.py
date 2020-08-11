""" Environment interface with custom sampling and replication """
#  Copyright (c) 2020, - All Rights Reserved
#  This file is part of the Evolutionary Planning on a Learned World Model thesis.
#  Unauthorized copying of this file, via any medium is strictly prohibited without the consensus of the authors.
#  Written by Thor V.A.N. Olesen <thorolesen@gmail.com> & Dennis T.T. Nguyen <dennisnguyen3000@yahoo.dk>.

import random
import gym
import numpy as np
import multiprocessing
from multiprocessing import Queue, Process
import copy
import uuid


class EnvironmentWrapper:
    def __init__(self, config, seed=None, action_buffer=None):
        self.seed = seed
        self.config = config
        self.environment = None
        self.game_name = config['game']
        self.is_discretize_sampling = config['planning']['is_discretize_sampling']

        action_buffer = [] if action_buffer is None else action_buffer  # Mutability check
        self.action_buffer = action_buffer

        self.cores = multiprocessing.cpu_count()
        self.env_processess = []

        self.min_replication_buffer_size = 16

    def step(self, action, is_simulation=False):
        if self.environment is None:
            raise Exception('Cannot call step before reset.')
        if not is_simulation:
            self.action_buffer.append(action)
        return self.environment.step(action)

    def reset(self, seed=None):
        if self.environment is None:
            self.environment = gym.make(self.game_name)

        self.seed = seed if seed else random.randint(0, 2 ** 31 - 1)
        self.environment.seed(self.seed)
        return self.environment.reset()

    def render(self):
        self.environment.render()

    def sample(self):
        return self._continous_sample() if not self.is_discretize_sampling else self._discrete_sample()

    def close(self):
        self.environment.close()

    def _continous_sample(self):
        steer = np.random.uniform(low=-1, high=1)
        gas = np.random.uniform(low=0, high=1)
        brake = np.random.uniform(low=0, high=1)
        return [steer, gas, brake]

    def _discrete_sample(self):
        steer_steps = np.arange(start=-1.00, stop=1.00, step=0.1)
        gas_steps = np.arange(start=0.00, stop=1.00, step=0.2)
        brake_steps = np.arange(start=0.00, stop=1.00, step=0.2)
        steer = np.random.choice(steer_steps)
        gas = np.random.choice(gas_steps)
        brake = np.random.choice(brake_steps)
        action = [steer, gas, brake]
        return (action)

    # State replications
    def replicate(self, is_render=False):
        environment = EnvironmentWrapper(self.config, action_buffer=copy.copy(self.action_buffer))
        environment.reset(self.seed)
        rewards = 0
        state = None
        for replay_action in self.action_buffer:
            state, reward, _, _ = environment.step(replay_action, is_simulation=True)
            rewards += reward
            if is_render:
                environment.render()
        return environment, rewards, state

    def mass_replicate(self, workers=None):
        workers = workers if workers else self.cores
        for i in range(workers):
            replica = Replication('CarRacing-v0', seed=self.seed, action_history=self.action_buffer)
            agent_actions, output = Queue(), Queue()
            process = Process(name=f'thread_{len(self.env_processess)}', target=start, args=(replica, agent_actions, output))
            process.start()
            self.env_processess.append((process, agent_actions, output))

    def clear_mass_replicate(self):
        while len(self.env_processess) is not 0:
            process, agent_actions, output = self.env_processess.pop()
            process.terminate()
            self._clear_replica_thread(agent_actions, output)

    def get_reward_from_buffered_environment(self, action_sequence, macro_actions):
        if len(self.env_processess) == 0:
            self.mass_replicate(1)
        process, agent_actions, output = self.env_processess.pop()
        agent_actions.put((action_sequence, macro_actions))
        total_reward = output.get(block=True)

        return total_reward

    def _clear_replica_thread(self, agent_actions, output, process=None):
        agent_actions.close()
        agent_actions.join_thread()
        output.close()
        output.join_thread()
        if process:
            process.terminate()
    ######


class Replication:
    def __init__(self, game, seed, action_history):
        self.game = game
        self.environment = None
        self.seed = seed
        self.action_history = action_history
        self.id = uuid.uuid4()

    def create_track(self):
        self.environment = gym.make(self.game)
        self.environment.seed(self.seed)
        self.environment.reset()

    def replay_buffer(self):
        for action in self.action_history:
            self.environment.step(action)

    def step(self, action):
        _, reward, is_done, _ = self.environment.step(action)
        return reward, is_done

    def start(replication, agent_actions, output):
        replication.create_track()
        replication.replay_buffer()
        actions, macro_actions = agent_actions.get(block=True)
        total_reward = 0
        for action in actions:
            for _ in range(macro_actions):
                reward, is_done = replication.step(action)
                total_reward += reward
                if is_done:
                    break
        output.put(total_reward)




