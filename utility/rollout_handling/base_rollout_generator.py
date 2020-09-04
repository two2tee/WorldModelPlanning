import os
import platform

import numpy as np
from PIL import Image
from os.path import exists, join
from torch import multiprocessing
from torch.multiprocessing import Pool
from environment.environment_factory import get_environment
from environment.actions.action_sampler_factory import get_action_sampler

if platform.system() == "Darwin" or platform.system() == "Linux":
    print("Spawn method enabled over fork on Mac OSX / Linux")
    multiprocessing.set_start_method("spawn", force=True)


class BaseRolloutGenerator:
    def __init__(self, config, data_output_dir):
        self.data_dir = data_output_dir
        if not exists(self.data_dir):
            os.mkdir(self.data_dir)
        self.config = config
        self.action_sampler = get_action_sampler(self.config)
        self.rollouts = config["data_generator"]['rollouts']
        self.sequence_length = config["data_generator"]['sequence_length']
        self.threads = multiprocessing.cpu_count()
        self.render_mode = False
        self.img_width = 64
        self.img_height = 64
        print(f'data_dir: {self.data_dir} | cores: {self.threads}')

    def generate_rollouts(self):
        self.threads = self.rollouts if self.rollouts < self.threads else self.threads
        rollouts_per_thread = int(self.rollouts / self.threads)
        print(f'{self.rollouts} rollouts across {self.threads} cores - {rollouts_per_thread} rollouts per thread.')
        with Pool(self.threads) as pool:
            threads = [pool.apply_async(self._run_rollout_thread, args=(rollouts_per_thread, thread))
                       for thread in range(1, self.threads + 1)]
            [thread.get() for thread in threads]
            pool.close()

        print(f'Done - {self.rollouts} rollout samples for {self.config["game"]} saved in {self.data_dir}')

    def _run_rollout_thread(self, rollouts, thread):
        i = 1
        environment = get_environment(self.config)
        while i < rollouts + 1:
            actions_rollout, states_rollout, reward_rollout, is_done_rollout = self._standard_rollout(environment, thread, i, rollouts)

            if len(actions_rollout) < self.sequence_length:  # ensure rollouts contains enough data for sequence
                print(f'thread: {thread} - Bad rollout with {len(actions_rollout)} actions - retry...')
                i -= 1
            else:
                self._save_rollout(thread, i, states_rollout, reward_rollout, actions_rollout, is_done_rollout)
            i += 1

        return thread

    def _standard_rollout(self, environment, thread, current_rollout, rollouts):
        return NotImplemented

    def _step(self, environment, obs, previous_action, model=None):
        return NotImplemented

    def _reset(self, environment):
        return NotImplemented

    def _compress_frame(self, frame, is_resize=False):
        if is_resize:
            frame = np.array(Image.fromarray(frame).resize(size=(self.img_width, self.img_height)))
        return frame

    def _save_rollout(self, thread, rollout_number, states_rollout, reward_rollout, actions_rollout, is_done_rollout):
        print(f"Thread {thread} - End of rollout {rollout_number}, {len(states_rollout)} frames.")
        print(self.data_dir, f'{self.config["game"]}_thread_{thread}_rollout_{rollout_number}')
        np.savez_compressed(file=join(self.data_dir,
                                      f'{self.config["game"]}{self.config["data_generator"]["data_prefix"]}thread_{thread}_rollout_{rollout_number}'),
                            observations=np.array(states_rollout),
                            rewards=np.array(reward_rollout),
                            actions=np.array(actions_rollout),
                            terminals=np.array(is_done_rollout))