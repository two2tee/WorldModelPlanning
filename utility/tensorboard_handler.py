""" Tensorboard interface to enable logging of MDRNN/VAE training data """
#  Copyright (c) 2020, - All Rights Reserved
#  This file is part of the Evolutionary Planning on a Learned World Model thesis.
#  Unauthorized copying of this file, via any medium is strictly prohibited without the consensus of the authors.
#  Written by Thor V.A.N. Olesen <thorolesen@gmail.com> & Dennis T.T. Nguyen <dennisnguyen3000@yahoo.dk>.
import io
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
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

    def log_average_loss_per_epoch(self, name, loss, epoch, is_train):
        if not self.is_logging:
            return
        title = f"{name} - Average Total Loss/per_epoch"
        self.log_train_test(title, loss, epoch, is_train)

    def log_reward_loss_per_epoch(self, name, loss, epoch, is_train):
        title = f"{name} - Reward MSE Loss/per_epoch"
        self.log_train_test(title, loss, epoch, is_train)

    def log_baseline_reward_loss_per_epoch(self, name, loss, epoch, is_train):
        title = f"{name} - Reward Baseline MSE Loss/per_epoch"
        self.log_train_test(title, loss, epoch, is_train)

    def log_terminal_loss_per_epoch(self, name, loss, epoch, is_train):
        title = f"{name} - Terminals BCE Loss/per_epoch"
        self.log_train_test(title, loss, epoch, is_train)

    def log_next_latent_loss_per_epoch(self, name, loss, epoch, is_train):
        title = f"{name} - GMM next latent Loss/per_epoch"
        self.log_train_test(title, loss, epoch, is_train)

    def log_average_loss_per_batch(self, name, loss, batch, is_train):
        title = f"{name} - Average Total Loss/per_batch"
        self.log_train_test(title, loss, batch, is_train)

    def log_reward_loss_per_batch(self, name, loss, batch, is_train):
        title = f"{name} - Reward MSE Loss/per_batch"
        self.log_train_test(title, loss, batch, is_train)

    def log_baseline_reward_loss_per_batch(self, name, loss, batch, is_train):
        title = f"{name} - Reward Baseline MSE Loss/per_batch"
        self.log_train_test(title, loss, batch, is_train)

    def log_reward_baseline_value(self, name, model, baseline_reward, num_files):
        self._train_writer.add_text(tag=f'{name} - baseline_reward/{model}', text_string=f'Baseline reward: {baseline_reward}\n'
                                                                                         f'Reward count: {num_files}\n')

    def log_reward_baseline_losses(self, name, model, train_loss, test_loss):
        self._train_writer.add_text(tag=f'{name} - baseline_reward/{model}', text_string=f'Baseline train loss: {train_loss}\n'
                                                                                         f'Baseline test loss: {test_loss}')

    def log_terminal_loss_per_batch(self, name, loss, batch, is_train):
        title = f"{name} - Terminals BCE Loss/per_batch"
        self.log_train_test(title, loss, batch, is_train)

    def log_next_latent_loss_per_batch(self, name, loss, batch, is_train):
        title = f"{name} - GMM next latent Loss/per_batch"
        self.log_train_test(title, loss, batch, is_train)

    def log_vae_reconstruction(self, images, epoch):
        if not self.is_logging:
            return
        grid = make_grid(images)
        self._train_writer.add_image(f'reconstruction_{epoch}', grid, 0)

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

    def log_batch_sample(self, samples, batch_idx):
        # Save the plot to a PNG in memory and prevent display

        current_frame_subplot = 1
        num_samples = len(samples['input_frames'])

        figure = plt.figure(figsize=(15, 15))
        for i in range(num_samples):
            self.add_to_image_grid('Input\n'
                                   f'action: {list(samples["input_actions"][i])}\n', current_frame_subplot, samples['input_frames'][i], num_samples)
            current_frame_subplot += 1
            self.add_to_image_grid(f'Prediction\n'
                                   f'reward_pred: {samples["pred_rewards"][i].item()}\n'
                                   f'terminal_pred: {samples["pred_terminals"][i].item()}',
                                   current_frame_subplot, samples['pred_frames'][i], num_samples)
            current_frame_subplot += 1
            self.add_to_image_grid('Target\n'
                                   f'reward_target: {samples["target_rewards"][i].item()}\n'
                                   f'terminal_target: {samples["target_terminals"][i].item()}', current_frame_subplot, samples['target_frames'][i], num_samples)
            current_frame_subplot += 1

        image = self.plot_to_image(figure)
        self._train_writer.add_image(f"Batch_train_samples", image, dataformats='HWC', global_step=batch_idx)
        self.commit_log()

    def plot_to_image(self, figure):
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        pil_img = Image.open(buf).convert('RGB')
        image = np.array(pil_img)
        return image

    def add_to_image_grid(self, title, index, image, num_samples):
        plt.subplot(num_samples, 3, index, title=title)  # 3 = input, pred, target
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image)

    # def log_standard_planning_test(self, test_name, agent, result):
    #     self._planning_test_writer = SummaryWriter(log_dir=f'{self.log_dir_root}/planning_test/{name}')

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


