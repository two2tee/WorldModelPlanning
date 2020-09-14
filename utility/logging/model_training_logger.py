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
from utility.logging.base_logger import BaseLogger


class ModelTrainingLogger(BaseLogger):

    def __init__(self, is_logging):
        super().__init__(is_logging)
        self._test_writer = None
        self._train_writer = None
        self.start_time = None

    def start_log_training(self, name, model, dataloader, save_input_image=True):
        if not self._is_logging:
            return
        dataiter = iter(dataloader)
        frames = dataiter.next()
        self.start_log(name)
        if save_input_image:
            grid_of_images = make_grid(frames)
            self._train_writer.add_image(tag=f'{name}_training_input_example', img_tensor=grid_of_images, global_step=0)
        self._train_writer.add_graph(model, frames)
        self.start_time = time.time()

    def start_log_training_minimal(self, name,  is_vae=False):
        if not self._is_logging:
            return
        self.start_log(name, is_vae)
        self.start_time = time.time()

    def log_average_loss_per_epoch(self, name, loss, epoch, is_train):

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

    def log_vae_random_constructions(self, images, epoch):
        if not self._is_logging:
            return
        grid = make_grid(images)
        self._test_writer.add_image(f'vae_random_latent_vectors_construction', grid, global_step=epoch)

    def log_vae_reconsstructions(self, targets, predictions, epoch, is_train):
        if not self._is_logging:
            return
        target_grid = make_grid(targets)
        predictions = make_grid(predictions)

        if is_train:
            self._train_writer.add_image(f'vae_reconstructions/train_targets', target_grid, global_step=epoch)
            self._train_writer.add_image(f'vae_reconstructions/train_predictions', predictions, global_step=epoch)
        else:
            self._test_writer.add_image(f'vae_reconstructions/test_targets', target_grid, global_step=epoch)
            self._test_writer.add_image(f'vae_reconstructions/test_predictions', predictions, global_step=epoch)

    def log_train_test(self, tag, loss, epoch, is_train):
        if not self._is_logging:
            return
        if is_train:
            self._train_writer.add_scalar(tag, loss, epoch)
        else:
            self._test_writer.add_scalar(tag, loss, epoch)

    def log_batch_sample(self, samples, batch_idx, is_train):
        # Save the plot to a PNG in memory and prevent display
        image = self._make_batch_image_grid(samples)
        if is_train:
            self._train_writer.add_image(f"Batch_train_samples", image, dataformats='HWC', global_step=batch_idx)
        else:
            self._test_writer.add_image(f"Batch_test_samples", image, dataformats='HWC', global_step=batch_idx)
        self.commit_log()

    def _make_batch_image_grid(self, samples):
        current_frame_subplot = 1
        num_samples = len(samples['input_frames'])

        figure = plt.figure(figsize=(15, 15))
        for i in range(num_samples):
            self._add_to_image_grid('Input\n'
                                   f'action: {list(samples["input_actions"][i])}\n', current_frame_subplot, samples['input_frames'][i], num_samples)
            current_frame_subplot += 1
            self._add_to_image_grid(f'Prediction\n'
                                   f'reward_pred: {samples["pred_rewards"][i].item()}\n'
                                   f'terminal_pred: {samples["pred_terminals"][i].item()}',
                                   current_frame_subplot, samples['pred_frames'][i], num_samples)
            current_frame_subplot += 1
            self._add_to_image_grid('Target\n'
                                   f'reward_target: {samples["target_rewards"][i].item()}\n'
                                   f'terminal_target: {samples["target_terminals"][i].item()}', current_frame_subplot, samples['target_frames'][i], num_samples)
            current_frame_subplot += 1

        return self._plot_to_image(figure)

    def _plot_to_image(self, figure):
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(figure)
        buf.seek(0)
        pil_img = Image.open(buf).convert('RGB')
        image = np.array(pil_img)
        return image

    def _add_to_image_grid(self, title, index, image, num_samples):
        plt.subplot(num_samples, 3, index, title=title)  # 3 = input, pred, target
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image)

    def end_log_training(self, name):
        if not self._is_logging:
            return
        elapsed_time = (time.time() - self.start_time)/60
        self._test_writer.add_text(tag=f'{name} - Total train time', text_string=f'Minutes: {elapsed_time}')
        self.end_log()

    def start_log(self, name, is_vae=False):
        if not self._is_logging:
            return
        self._test_writer = SummaryWriter(log_dir=f'{self.log_dir_root}/test/{name}')
        self._train_writer = SummaryWriter(log_dir=f'{self.log_dir_root}/train/{name}')

    def commit_log(self):
        if not self._is_logging:
            return
        self._train_writer.flush()
        self._test_writer.flush()

    def end_log(self):
        if not self._is_logging:
            return
        self.commit_log()
        self._train_writer.close()
        self._test_writer.close()
