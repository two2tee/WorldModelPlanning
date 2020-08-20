""" Tensorboard interface to enable logging of MDRNN/VAE training data """
#  Copyright (c) 2020, - All Rights Reserved
#  This file is part of the Evolutionary Planning on a Learned World Model thesis.
#  Unauthorized copying of this file, via any medium is strictly prohibited without the consensus of the authors.
#  Written by Thor V.A.N. Olesen <thorolesen@gmail.com> & Dennis T.T. Nguyen <dennisnguyen3000@yahoo.dk>.

import time
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter


class TensorboardHandler:
    def __init__(self, is_logging):
        self.log_dir_root = 'utility/tensorboard_runs'
        self._test_writer = None
        self._train_writer = None
        self._planning_test_writer = None
        self.start_time = None
        self.is_logging = is_logging

    def start_log_training(self, name, model, dataloader, save_input_image=True):
        if not self.is_logging:
            return
        dataiter = iter(dataloader)
        frames = dataiter.next()
        self.start_log(name)
        if save_input_image:
            grid_of_images = make_grid(frames)
            self._train_writer.add_image(tag=f'{name}_training_input_example', img_tensor=grid_of_images, global_step=0)
        self._train_writer.add_graph(model, frames)
        self.start_time = time.time()

    def start_log_training_minimal(self, name):
        if not self.is_logging:
            return
        self.start_log(name)
        self.start_time = time.time()

    def log_average_loss(self, name, loss, epoch, is_train):
        if not self.is_logging:
            return
        title = f"{name} - Average Total Loss"
        self.log_train_test(title, loss, epoch, is_train)

    def log_reward_loss(self, name, loss, epoch, is_train):
        title = f"{name} - Reward MSE Loss"
        self.log_train_test(title, loss, epoch, is_train)

    def log_terminal_loss(self, name, loss, epoch, is_train):
        title = f"{name} - Terminals BCE Loss"
        self.log_train_test(title, loss, epoch, is_train)

    def log_next_latent_loss(self, name, loss, epoch, is_train):
        title = f"{name} - GMM next latent Loss"
        self.log_train_test(title, loss, epoch, is_train)

    def log_vae_reconstruction(self, images, epoch):
        if not self.is_logging:
            return
        grid = make_grid(images)
        self._train_writer.add_image(f'reconstruction_{epoch}', grid, 0)

    def end_log_training(self, name):
        if not self.is_logging:
            return
        elapsed_time = (time.time() - self.start_time)/60
        self._test_writer.add_text(tag=f'{name} - Total train time', text_string=f'Minutes: {elapsed_time}')
        self.end_log()

    def start_log(self, name):
        if not self.is_logging:
            return
        self._test_writer = SummaryWriter(log_dir=f'{self.log_dir_root}/test/{name}')
        self._train_writer = SummaryWriter(log_dir=f'{self.log_dir_root}/train/{name}')
        self._planning_test_writer = SummaryWriter(log_dir=f'{self.log_dir_root}/planning_test/{name}')

    def commit_log(self):
        self._train_writer.flush()
        self._test_writer.flush()
        self._planning_test_writer.flush()

    def end_log(self):
        self.commit_log()
        self._train_writer.close()
        self._test_writer.close()
        self._planning_test_writer.close()


    def log_train_test(self, tag, loss, epoch, is_train):
        if is_train:
            self._train_writer.add_scalar(tag, loss, epoch)
        else:
            self._test_writer.add_scalar(tag, loss, epoch)

    def log_iteration_max_reward(self, name, iteration, max_reward):
        title = f"{name}/Average Max reward"
        self._planning_test_writer.add_scalar(title, max_reward, iteration)

    def log_iteration_avg_reward(self, name, iteration, avg_reward):
        title = f"{name}/Average Total reward"
        self._planning_test_writer.add_scalar(title, avg_reward, iteration)