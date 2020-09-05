import json

from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

from utility.rollout_handling.mdrnn_loaders import RolloutSequenceDataset

with open('../../config.json') as config_file:
    config = json.load(config_file)


def transform(frames):
    # 0=batch size, 3=img channels, 1 and 2 = img dims, / 255 normalize
    transform = transforms.Lambda(lambda img: np.transpose(img, (0, 3, 1, 2)) / 255)
    return transform(frames)


data_dir = config['data_dir']



def create_loader(is_train=True, is_same_test_data=False,
                  random_sampling=False, batch_size=16, file_ratio=0.8, buffer_size=30,
                  shuffle=False, drop_last=False):
    dataset = RolloutSequenceDataset(root=data_dir,
                                     seq_len=500,
                                     transform=transform,
                                     is_train=is_train,
                                     is_same_testdata=is_same_test_data,
                                     buffer_size=buffer_size,
                                     file_ratio=file_ratio,
                                     max_size=0, is_random_sampling=random_sampling)

    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        num_workers=0,
                        shuffle=shuffle,
                        drop_last=drop_last)

    return loader

def extract(batch):
    obs, actions, rewards, terminals, next_obs = batch
    return obs, actions, rewards, terminals, next_obs

def no_shuffle_same_batch_on_new_iter():
    loader = create_loader(shuffle=False)
    batches = iter(loader)
    obs, actions, rewards, terminals, next_obs = extract(next(batches))
    a = actions[0].numpy()
    batches = iter(loader)
    obs, actions, rewards, terminals, next_obs = extract(next(batches))
    b = actions[0].numpy()
    assert np.array_equal(a, b)


def shuffle_different_batch_on_new_iter():
    loader = create_loader(shuffle=True)
    batches = iter(loader)
    obs, actions, rewards, terminals, next_obs = extract(next(batches))
    a = actions[0].numpy()
    batches = iter(loader)
    obs, actions, rewards, terminals, next_obs = extract(next(batches))
    b = actions[0].numpy()
    assert not np.array_equal(a, b)

no_shuffle_same_batch_on_new_iter()
shuffle_different_batch_on_new_iter()