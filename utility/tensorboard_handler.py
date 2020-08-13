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
        self.writer = None
        self.start_time = None
        self.is_logging = is_logging

    def start_log_training(self, name, model, dataloader, save_input_image=True):
        if not self.is_logging:
            return
        dataiter = iter(dataloader)
        frames = dataiter.next()
        self.writer = SummaryWriter(f'{self.log_dir_root}/{name}')
        if save_input_image:
            grid_of_images = make_grid(frames)
            self.writer.add_image(tag=f'{name}_training_input_example', img_tensor=grid_of_images, global_step=0)
        self.writer.add_graph(model, frames)
        self.start_time = time.time()

    def start_log_training_minimal(self, name):
        if not self.is_logging:
            return
        self.writer = SummaryWriter(f'{self.log_dir_root}/{name}')
        self.start_time = time.time()

    def log_average_loss(self, name, loss, epoch, is_train):
        if not self.is_logging:
            return
        title = f"{name} - Average Loss/{'train' if is_train else 'test'}"
        self.writer.add_scalar(title, loss, epoch)

    def log_reward_loss(self, name, loss, epoch, is_train):
        title = f"{name} - Reward Loss/{'train' if is_train else 'test'}"
        self.writer.add_scalar(title, loss, epoch)

    def log_terminal_loss(self, name, loss, epoch, is_train):
        title = f"{name} - Terminals Loss/{'train' if is_train else 'test'}"
        self.writer.add_scalar(title, loss, epoch)

    def log_next_latent_loss(self, name, loss, epoch, is_train):
        title = f"{name} - next latent Loss/{'train' if is_train else 'test'}"
        self.writer.add_scalar(title, loss, epoch)

    def log_vae_reconstruction(self, images, epoch):
        if not self.is_logging:
            return
        grid = make_grid(images)
        self.writer.add_image(f'reconstruction_{epoch}', grid, 0)

    def end_log_training(self, name):
        if not self.is_logging:
            return
        elapsed_time = (time.time() - self.start_time)/60
        self.writer.add_text(tag=f'{name} - Total train time', text_string=f'Minutes: {elapsed_time}')
        self.writer.flush()
        self.writer.close()
