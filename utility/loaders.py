"""Data loader used by pytorch to load rollout data - based on https://github.com/ctallec/world-models"""

from bisect import bisect
import os
# import random
import numpy as np
import torch.utils.data
from tqdm import tqdm


# TODO: pretrained RL agent rollouts for testing only (later use planning rollouts only to split)
# TODO: 80/20 train test ratio?, shuffle or not?
# Original code by Ctallec: https://github.com/ctallec/world-models/blob/master/data/loaders.py
class _RolloutDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, buffer_size=100, is_train=True, file_ratio=0.5):
        self._transform = transform
        self._files = [os.path.join(root, name) for root, dirs, files in os.walk(root) for name in files]

        # self._files = self._files[:-total_num_files] if is_train else self._files[-total_num_files:]
        self._files = self.take_x_ratio_files(is_train, file_ratio)

        self._buffer_index = 0
        self._buffer_size = buffer_size
        self._cum_size, self._buffer, self._buffer_file_names = None, None, None

        if len(self._files) < self._buffer_size:
            raise Exception(f"Too low number of files {len(self._files)} with larger buffer of size: {self._buffer_size}")

    def take_x_ratio_files(self, is_train, ratio):
        to_take_train = int(len(self._files) * ratio)
        to_take_test = len(self._files) - to_take_train
        return self._files[:-to_take_train] if is_train else self._files[-to_take_test:]

    def load_next_buffer(self):
        """ Loads next buffer """
        self._buffer_fnames = self._files[self._buffer_index:self._buffer_index + self._buffer_size]
        self._buffer_index += self._buffer_size
        self._buffer_index = self._buffer_index % len(self._files)
        self._buffer = []
        self._cum_size = [0]

        # progress bar
        pbar = tqdm(total=len(self._buffer_fnames),
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
        pbar.set_description("Loading file buffer ...")

        for f in self._buffer_fnames:
            with np.load(f) as data:
                self._buffer += [{k: np.copy(v) for k, v in data.items()}]
                self._cum_size += [self._cum_size[-1] + self._data_per_sequence(data['rewards'].shape[0])]

            pbar.update(1)
        pbar.close()

    def __len__(self):
        # to have a full sequence, you need self.seq_len + 1 elements, as
        # you must produce both an seq_len obs and seq_len next_obs sequences
        if not self._cum_size:
            self.load_next_buffer()
        return self._cum_size[-1]

    def __getitem__(self, i):
        # binary search through cum_size
        file_index = bisect(self._cum_size, i) - 1
        seq_index = i - self._cum_size[file_index]
        data = self._buffer[file_index]
        return self._get_data(data, seq_index)

    def _get_data(self, data, seq_index):
        pass

    def _data_per_sequence(self, data_length):
        pass


class RolloutSequenceDataset(_RolloutDataset):
    """ Encapsulates rollouts.
    Rollouts should be stored in subdirs of the root directory, in the form of npz files,
    each containing a dictionary with the keys:
        - observations: (rollout_len, *obs_shape)
        - actions: (rollout_len, action_size)
        - rewards: (rollout_len,)
        - terminals: (rollout_len,), boolean
     As the dataset is too big to be entirely stored in rams, only chunks of it
     are stored, consisting of a constant number of files (determined by the
     buffer_size parameter).  Once built, buffers must be loaded with the
     load_next_buffer method.
    Data are then provided in the form of tuples (obs, action, reward, terminal, next_obs):
    - obs: (seq_len, *obs_shape)
    - actions: (seq_len, action_size)
    - reward: (seq_len,)
    - terminal: (seq_len,) boolean
    - next_obs: (seq_len, *obs_shape)
    NOTE: seq_len < rollout_len in most use cases
    :args root: root directory of data_random_car sequences
    :args seq_len: number of timesteps extracted from each rollout
    :args transform: transformation of the observations
    :args train: if True, train data_random_car, else test
    """
    def __init__(self, root, seq_len, transform, buffer_size=100, is_train=True, file_ratio=0.5):
        super().__init__(root, transform, buffer_size, is_train, file_ratio)
        self._seq_len = seq_len

    def _get_data(self, data, seq_index):
        obs_data = data['observations'][seq_index:seq_index + self._seq_len + 1]
        obs_data = self._transform(obs_data.astype(np.float32))
        obs, next_obs = obs_data[:-1], obs_data[1:]  # TEACHER FORCING
        action = data['actions'][seq_index+1:seq_index + self._seq_len + 1]
        action = action.astype(np.float32)
        reward, terminal = [data[key][seq_index+1:seq_index + self._seq_len + 1].astype(np.float32)
                            for key in ('rewards', 'terminals')]

        return obs, action, reward, terminal, next_obs  # data_random_car format

    def _data_per_sequence(self, data_length):
        # return data_length
        if data_length < self._seq_len:
            raise Exception(f'Sequence length in data is {data_length} which less than stated sequence length of {self._seq_len}')
        return data_length - self._seq_len


class RolloutObservationDataset(_RolloutDataset):
    """ Encapsulates rollouts.
    Rollouts should be stored in subdirs of the root directory, in the form of npz files,
    each containing a dictionary with the keys:
        - observations: (rollout_len, *obs_shape)
        - actions: (rollout_len, action_size)
        - rewards: (rollout_len,)
        - terminals: (rollout_len,), boolean
     As the dataset is too big to be entirely stored in rams, only chunks of it
     are stored, consisting of a constant number of files (determined by the
     buffer_size parameter).  Once built, buffers must be loaded with the
     load_next_buffer method.
    Data are then provided in the form of images
    :args root: root directory of data_random_car sequences
    :args seq_len: number of timesteps extracted from each rollout
    :args transform: transformation of the observations
    :args train: if True, train data_random_car, else test
    """
    def _data_per_sequence(self, data_length):
        return data_length

    def _get_data(self, data, seq_index):
        return self._transform(data['observations'][seq_index])
