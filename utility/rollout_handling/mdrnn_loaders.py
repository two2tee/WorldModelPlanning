"""Data loader used by pytorch to load rollout data - based on https://github.com/ctallec/world-models"""

import os
import random
import numpy as np
import torch.utils.data

# Original code by Ctallec: https://github.com/ctallec/world-models/blob/master/data/loaders.py
class _MDRNNRolloutDataset(torch.utils.data.Dataset):
    CURRENT_TESTDATA = set()

    def __init__(self, root, transform, buffer_size=100, is_train=True, is_same_testdata=False, file_ratio=0.5, max_size=0, is_random_sampling=False):
        self.is_same_testdata = is_same_testdata
        self._transform = transform
        self._files = [os.path.join(root, name) for root, dirs, files in os.walk(root) for name in files]

        if is_same_testdata and len(_MDRNNRolloutDataset.CURRENT_TESTDATA):
            self._take_last_as_test(self._files, max_size, file_ratio) # TODO Since we use HA data for initial tests we want to always use same tests here and never random rollouts in ha
        elif not is_train and not is_same_testdata:
            self._clear_test_data()

        self._files = self._random_sampling(self._files, max_size, is_train, file_ratio) if is_random_sampling else \
                      self._standard_sampling(self._files, max_size, is_train, file_ratio)

        if not is_train:
            self._set_current_testdata(self._files)

        self._buffer_index = 0
        self._buffer_size = buffer_size
        self._buffer = None

        if len(self._files) < self._buffer_size:
            raise Exception(f"Too low number of files {len(self._files)} with larger buffer of size: {self._buffer_size}")

    def _standard_sampling(self, files, max_size, is_train, file_ratio):
        files_to_take = max_size if len(files) >= max_size > 0 else len(files)
        files = files[:files_to_take] if max_size > 0 else files
        train_ratio, test_ratio = self._calc_file_ratio(len(files), file_ratio)
        return files[:train_ratio] if is_train else files[-test_ratio:]

    def _random_sampling(self, files, max_size, is_train, file_ratio):
        files_to_take = max_size if len(files) >= max_size > 0 else len(files)
        train_ratio, test_ratio = self._calc_file_ratio(files_to_take, file_ratio)
        files_to_take = train_ratio if is_train else test_ratio
        sampled_files = []

        while len(sampled_files) < files_to_take:
            potential_file = random.choice(files)
            if potential_file not in _MDRNNRolloutDataset.CURRENT_TESTDATA and potential_file not in sampled_files:
                sampled_files.append(potential_file)
        return sampled_files


    def _take_last_as_test(self, files, max_size, file_ratio):
        files_to_take = max_size if len(files) >= max_size > 0 else len(files)
        train_ratio, test_ratio = self._calc_file_ratio(files_to_take, file_ratio)
        return files[-test_ratio:]

    def _set_current_testdata(self, files):
        if self.is_same_testdata and len(_MDRNNRolloutDataset.CURRENT_TESTDATA) > 0:
            return

        self._clear_test_data()
        for file in files:
            _MDRNNRolloutDataset.CURRENT_TESTDATA.add(file)

    def _clear_test_data(self):
        _MDRNNRolloutDataset.CURRENT_TESTDATA = set()

    def _calc_file_ratio(self, num_files, file_ratio):
        train_ratio = int(num_files * file_ratio)
        test_ratio = num_files - train_ratio
        return train_ratio, test_ratio

    def __len__(self):
        return len(self._files)

    def __getitem__(self, i):
        # print(f'loaded: {self._files[i]}')
        with np.load(self._files[i]) as data:
            data = {k: np.copy(v) for k, v in data.items()}
        return self._get_data(data)

    def _get_data(self, data):
        pass

    def _data_per_sequence(self, data_length):
        pass


class RolloutSequenceDataset(_MDRNNRolloutDataset):
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
    def __init__(self, root, seq_len, transform, buffer_size=100, is_train=True, is_same_testdata=False, file_ratio=0.5,
                       max_size=0, is_random_sampling=False):
        super().__init__(root, transform, buffer_size, is_train, is_same_testdata, file_ratio, max_size, is_random_sampling)
        self._seq_len = seq_len

    def _get_data(self, data):
        obs_data = data['observations'][:self._seq_len + 1] if self._seq_len else data['observations'][:]
        obs_data = self._transform(obs_data.astype(np.float32))
        obs, next_obs = obs_data[:-1], obs_data[1:]
        action = data['actions'][:self._seq_len] if self._seq_len else data['actions'][:]
        action = action.astype(np.float32)
        reward, terminal = [data[key][:self._seq_len].astype(np.float32) for key in ('rewards', 'terminals')] if self._seq_len else \
                           [data[key][:].astype(np.float32) for key in ('rewards', 'terminals')]

        return obs, action, reward, terminal, next_obs  # data_random_car format

    def _data_per_sequence(self, data_length):
        if data_length < self._seq_len:
            raise Exception(f'Sequence length in data is {data_length} which less than stated sequence length of {self._seq_len}')
        return data_length - self._seq_len


