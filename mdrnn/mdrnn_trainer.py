""" MDRNN Trainer responsible for training and reloading MDRNNs
    Based on: https://github.com/ctallec/world-models/blob/master/trainmdrnn.py
"""
#  Copyright (c) 2020, - All Rights Reserved
#  This file is part of the Evolutionary Planning on a Learned World Model thesis.
#  Unauthorized copying of this file, via any medium is strictly prohibited without the consensus of the authors.
#  Written by Thor V.A.N. Olesen <thorolesen@gmail.com> & Dennis T.T. Nguyen <dennisnguyen3000@yahoo.dk>.

import torch
import platform
import numpy as np
import multiprocessing
import torch.nn.functional as f
from os import mkdir

from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torch import optim
from functools import partial
from os.path import join, exists
from torchvision import transforms
from torch.distributions import Normal, Categorical
from torch.utils.data import DataLoader
# from mdrnn.learning import ReduceLROnPlateau
from utility.loaders import RolloutSequenceDataset
from mdrnn.learning import EarlyStopping


def transform(frames):
    # 0=batch size, 3=img channels, 1 and 2 = img dims, / 255 normalize
    transform = transforms.Lambda(lambda img: np.transpose(img, (0, 3, 1, 2)) / 255)
    return transform(frames)


# Loss for gaussian mixture model
""" Computes the gmm loss.
    Compute minus the log probability of batch under the GMM model described
    by mus, sigmas, pi. Precisely, with bs1, bs2, ... the sizes of the batch
    dimensions (several batch dimension are useful when you have both a batch
    axis and a time step axis), gs the number of mixtures and fs the number of
    features.
    :args latent_next_obs: (bs1, bs2, *, fs) torch tensor
    :args mus: (bs1, bs2, *, gs, fs) torch tensor
    :args sigmas: (bs1, bs2, *, gs, fs) torch tensor
    :args logpi: (bs1, bs2, *, gs) torch tensor
    :args reduce: if not reduce, the mean in the following formula is ommited
"""
def gmm_loss(latent_next_obs, mus, sigmas, logpi, reduce=True):
    latent_next_obs = latent_next_obs.unsqueeze(-2)
    normal_dist = Normal(mus, sigmas)
    g_log_probs = normal_dist.log_prob(latent_next_obs)
    g_log_probs = logpi + torch.sum(g_log_probs, dim=-1)
    max_log_probs = torch.max(g_log_probs, dim=-1, keepdim=True)[0]
    g_log_probs = g_log_probs - max_log_probs
    g_probs = torch.exp(g_log_probs)
    probs = torch.sum(g_probs, dim=-1)

    log_prob = max_log_probs.squeeze() + torch.log(probs)
    if reduce:
        return - torch.mean(log_prob)
    return - log_prob


class MDRNNTrainer:
    def __init__(self, config, preprocessor, logger):
        self.config = config
        self.device = self._set_device()
        self.logger = logger
        self.preprocessor = preprocessor
        self.vae = None
        self.mdrnn = None

        self.session_name = 'mdrnn_'+self.config['experiment_name']
        self.model_dir = self.config['mdrnn_dir']
        self.data_dir = self.config['data_dir']
        self.test_data_dir = self.config['test_data_dir']
        self.backup_dir = join(self.model_dir, f"checkpoints/backups")
        self.best_mdrnn_filename = self.config['experiment_name'] + '_' + self.config["mdrnn_trainer"]["mdrnn_best_filename"]
        self.checkpoint_filename = self.config['experiment_name'] + '_' + self.config["mdrnn_trainer"]["mdrnn_checkpoint_filename"]
        self.is_use_specific_test_data = self.config['is_use_specific_test_data']
        self.is_baseline_reward_loss = self.config['mdrnn_trainer']['is_baseline_reward_loss']

        if not exists(self.backup_dir):
            mkdir(self.backup_dir)

        if not exists(self.model_dir):
            mkdir(self.model_dir)

        if self.is_use_specific_test_data and not exists(self.test_data_dir):
            raise Exception(f'use of specific test data was enabled but no folder "{self.test_data_dir}" was found')

        self.latent_size = self.config["latent_size"]
        self.batch_size = self.config["mdrnn_trainer"]["batch_size"]
        self.batch_train_idx = 0
        self.sequence_length = self.config["mdrnn_trainer"]["sequence_length"]
        self.N_train_batch_until_test_batch = self.config['mdrnn_trainer']['N_train_batch_until_test_batch']
        self.N_train_batch_until_pred_sampling = self.config['mdrnn_trainer']['N_train_batch_until_pred_sampling']
        self.is_random_sampling = self.config['mdrnn_trainer']['is_random_sampling']

        self.baseline_test_loss, self.baseline_train_loss = 0, 0

        self.num_workers = self.config['mdrnn_trainer']['num_workers']
        self.num_workers = self.num_workers if self.num_workers <= multiprocessing.cpu_count() else multiprocessing.cpu_count()

        self.optimizer = None
        self.scheduler = None
        self.earlystopping = None

        self.train_loader = None
        self.test_loader = None

        self.is_iterative = self.config["is_iterative_train_mdrnn"] and not self.config["is_train_mdrnn"]

    def train(self, vae, mdrnn, data_dir=None, max_epochs=None, seq_len=None, iteration=None, max_size=0, random_sampling=None):
        self.batch_train_idx = 0
        self.baseline_test_loss, self.baseline_train_loss = 0, 0
        self.sequence_length = self.config['mdrnn_trainer']['sequence_length'] if seq_len is None else seq_len
        self.data_dir = self.data_dir if data_dir is None else data_dir
        self.is_random_sampling = self.is_random_sampling if random_sampling is None else random_sampling
        self.vae = vae.to(self.device)
        self.mdrnn = mdrnn.to(self.device)
        self._load_data(max_size, is_random_sampling=self.is_random_sampling)
        self.logger.start_log_training_minimal(name=f'{self.session_name}{f"_iteration_{iteration}" if self.is_iterative else ""}')
        self.optimizer = optim.Adam(self.mdrnn.parameters(), lr=self.config['mdrnn_trainer']['learning_rate'])
        self.scheduler = self._get_scheduler(self.optimizer)
        self.earlystopping = EarlyStopping('min', patience=self.config['mdrnn_trainer']['early_stop_after_n_epochs'])
        train = partial(self._data_pass, is_train=True, include_reward=True)
        test = partial(self._data_pass, is_train=False, include_reward=True)

        start_epoch, current_best = self._reload_training_session()
        start_epoch += 1
        max_epochs = self.config['mdrnn_trainer']['max_epochs'] if max_epochs is None else max_epochs

        if self.is_baseline_reward_loss:
            avg_baseline_reward = self._calc_reward_avg_baseline()
            self.baseline_train_loss,  self.baseline_test_loss = self._calc_baseline_losses(avg_baseline_reward)

        if start_epoch > max_epochs:
            raise Exception(f'Inconsistent start epoch {start_epoch} and max_epoch {max_epochs}')

        self._load_data(max_size, is_random_sampling=random_sampling)
        test_losses = {}
        for epoch in range(start_epoch, max_epochs + 1):
            train(epoch)
            test_losses = test(epoch)
            self.scheduler.step(test_losses['average_loss'])
            self.earlystopping.step(test_losses['average_loss'])

            is_best = not current_best or test_losses['average_loss'] < current_best
            current_best = test_losses['average_loss'] if is_best else current_best

            self._save_checkpoint({'epoch': epoch,
                                   'batch_train_idx': self.batch_train_idx,
                                   'state_dict': self.mdrnn.state_dict(),
                                   'precision': test_losses['average_loss'], 'optimizer': self.optimizer.state_dict(),
                                   'scheduler': self.scheduler.state_dict(),
                                   'earlystopping': self.earlystopping.state_dict()
                                   }, is_best, iteration)
            if self.earlystopping.stop:
                print(f"End of Training because of early stopping at epoch {epoch}")
                break
        self.logger.end_log_training('mdrnn')
        return self.mdrnn, test_losses

    def reload_model(self, mdrnn, device = None):
        reload_file = join(self.model_dir, f'checkpoints/{self.best_mdrnn_filename}')
        if self.is_iterative:
            existing_iterative_file = join(self.model_dir, f'checkpoints/{"iterative_"}{self.best_mdrnn_filename}')
            reload_file = existing_iterative_file if exists(existing_iterative_file) else reload_file

        if not exists(reload_file):
            raise Exception('No MDRNN model found...')
        state = torch.load(reload_file, map_location=device if device else self.device)
        mdrnn.load_state_dict(state['state_dict'])
        print(f'Reloaded MDRNN model - {state["epoch"]}')
        return mdrnn

    def _reload_training_session(self):
        reload_file = join(self.model_dir, f"checkpoints/{'iterative_' if self.is_iterative else ''}{self.checkpoint_filename}")
        best_file = join(self.model_dir, f"checkpoints/{'iterative_' if self.is_iterative else ''}{self.best_mdrnn_filename}")

        reload_file = reload_file if exists(reload_file) and not self.is_iterative else best_file

        if exists(reload_file) and self.config['mdrnn_trainer']['is_continue_model']:
            state = torch.load(reload_file, map_location=self.device)
            best_test_loss = None
            if exists(best_file):
                best_state = torch.load(best_file, map_location=self.device)
                best_test_loss = best_state['precision']
                print(f"Reloading mdrnn at epoch {state['epoch']}, with best test error {best_test_loss} at epoch {best_state['epoch']}")
            else:
                print(f"Reloading mdrnn at epoch {state['epoch']}")

            self.batch_train_idx = state['batch_train_idx']
            self.mdrnn.load_state_dict(state["state_dict"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state['scheduler'])
            self.earlystopping.load_state_dict(state['earlystopping'])
            epoch = 1 if self.is_iterative else state['epoch']
            return epoch, best_test_loss
        print('No mdrnn found. Skip reloading...')
        return 0, None  # start epoch

    def _save_checkpoint(self, state, is_best, iteration):
        best_model_filename = join(self.model_dir, f"checkpoints/{'iterative_' if self.is_iterative else ''}{self.best_mdrnn_filename}")
        checkpoint_filename = join(self.model_dir, f"checkpoints/{'iterative_' if self.is_iterative else ''}{self.checkpoint_filename}")

        if self.is_iterative:  # For backup purpose
            iteration_filename = join(self.backup_dir, f"iterative_{iteration}_{self.best_mdrnn_filename}")
            torch.save(state, iteration_filename)

        torch.save(state, checkpoint_filename)
        if is_best or self.is_iterative:
            torch.save(state, best_model_filename)
            print(f'New best model found and saved')

    def _load_data(self, max_size=0, is_random_sampling=False):  # To avoid loading data when not training
        test_dataset = self._create_dataset(data_location=self.test_data_dir if self.is_use_specific_test_data else self.data_dir,
                                            buffer_size=self.config['mdrnn_trainer']['test_buffer_size'],
                                            file_ratio=self.config['mdrnn_trainer']['train_test_files_ratio'],
                                            is_train=False,
                                            is_same_testdata=self.is_use_specific_test_data,
                                            max_size=max_size,
                                            is_random_sampling=is_random_sampling
                                            )

        self.test_loader = DataLoader(dataset=test_dataset,
                                      batch_size=self.config['mdrnn_trainer']['batch_size'],
                                      num_workers=self.num_workers, shuffle=False, drop_last=True)

        train_dataset = self._create_dataset(data_location=self.data_dir,
                                             buffer_size=self.config['mdrnn_trainer']['train_buffer_size'],
                                             file_ratio=self.config['mdrnn_trainer']['train_test_files_ratio'],
                                             is_train=True,
                                             is_same_testdata=self.is_use_specific_test_data,
                                             max_size=max_size,
                                             is_random_sampling=is_random_sampling
                                             )

        self.train_loader = DataLoader(dataset=train_dataset,
                                       batch_size=self.config['mdrnn_trainer']['batch_size'],
                                       num_workers=self.num_workers, shuffle=False, drop_last=True)

    def _get_scheduler(self, optimizer):
        return ReduceLROnPlateau(optimizer,
                                 mode=self.config['mdrnn_trainer']['ReduceLROnPlateau']['mode'],
                                 factor=self.config['mdrnn_trainer']['ReduceLROnPlateau']['factor'],      # how much lr is reduced new_lr = lr * factor
                                 patience=self.config['mdrnn_trainer']['ReduceLROnPlateau']['patience'],  # How many epochs to ignore before lr changes
                                 verbose=self.config['mdrnn_trainer']['ReduceLROnPlateau']['print_lr_change'])

    def _set_device(self):
        if torch.cuda.is_available():
            if platform.system() == "Linux":
                print("GPU enabled on Linux machine...")
                torch.cuda.set_device(2)
            print(f'Cuda current device: {torch.cuda.current_device()}')
            return torch.device('cuda')
        print(f'Using CPU - Since no GPU was found or GPU driver outdated - ')
        return torch.device('cpu')

    def _create_dataset(self, data_location, buffer_size, file_ratio, is_train,is_same_testdata, max_size=0, is_random_sampling=False):
        dataset = RolloutSequenceDataset(root=data_location,
                                      seq_len=self.sequence_length,
                                      transform=transform,
                                      is_train=is_train,
                                      is_same_testdata=is_same_testdata,
                                      buffer_size=buffer_size,
                                      file_ratio=file_ratio,
                                      max_size=max_size, is_random_sampling=is_random_sampling)
        if len(dataset._files) == 0:
            raise Exception(f'No files found in {data_location}')
        return dataset

    def _data_pass(self, epoch, is_train, include_reward):
        cumulative_losses = {"loss": 0, "latent_loss": 0, "terminal_loss": 0, "reward_loss": 0}
        loader = self.train_loader if is_train else self.test_loader



        if is_train:
            cumulative_losses = self._train_epoch(epoch, cumulative_losses, loader, include_reward)
        else:
            cumulative_losses = self._test_epoch(epoch, cumulative_losses, loader, include_reward)

        loss, latent_loss, terminal_loss, reward_loss = self._extract_epoch_loss(cumulative_losses, loader)
        self._log_epoch_loss(loss, reward_loss, terminal_loss, latent_loss, self.baseline_train_loss, epoch, is_train=is_train)

        if len(loader.dataset) <= 0:
            print(f"Issue with data loader size {len(loader.dataset)}")
            return

        return {'average_loss': loss,
                'reward_loss': reward_loss,
                'terminal_loss': terminal_loss,
                'next_latent_loss': terminal_loss}

    def _train_epoch(self, epoch, cumulative_losses, loader, include_reward):
        self.mdrnn.train()
        batch_test_loader = iter(self.test_loader)
        last_tested_batch_idx = self.batch_train_idx

        progress_bar = tqdm(total=len(loader.dataset) if len(loader.dataset) >= 0 else 1, desc=f"Train Epoch {epoch}")
        for i, batch in enumerate(loader):
            latent_obs, actions, rewards, terminals, latent_next_obs = self._extract_batch_data(batch)

            if self._is_batch_testing():
                batch_test_loader = self._test_batch(batch_test_loader, include_reward)
                last_tested_batch_idx = self.batch_train_idx

            losses,  batch_results = self._train_step(latent_obs, actions, rewards, terminals, latent_next_obs, include_reward)

            self._update_cumulative_losses(cumulative_losses, current_losses=losses)
            batch_loss = losses['loss']
            batch_reward_loss = losses['mse']
            batch_terminal_loss = losses['bce']
            batch_latent_loss = losses['gmm'] / self.config['latent_size']

            self._log_batch_loss(batch_loss, batch_reward_loss, batch_terminal_loss, batch_latent_loss, self.baseline_train_loss, is_train=True)

            if self._loss_is_above_threshold(losses):
                print(f'Loss above threshold {losses}')

            if self._is_log_sample(losses):
                self._log_batch_prediction_results(batch_results)

            progress_bar.set_postfix_str(f"loss={batch_loss} bce={batch_terminal_loss} gmm={batch_latent_loss} mse={batch_reward_loss}")
            progress_bar.update(self.batch_size)
            self.batch_train_idx += 1

        progress_bar.close()

        if last_tested_batch_idx != self.batch_train_idx-1:
            self._test_batch(batch_test_loader, include_reward)

        return cumulative_losses

    def _test_epoch(self, epoch, cumulative_losses, loader, include_reward):
        self.mdrnn.eval()
        progress_bar = tqdm(total=len(loader.dataset) if len(loader.dataset) >= 0 else 1, desc=f"Test Epoch {epoch}")
        for i, batch in enumerate(loader):
            latent_obs, actions, rewards, terminals, latent_next_obs = self._extract_batch_data(batch)
            losses,  batch_results = self._test_step(latent_obs, actions, rewards, terminals, latent_next_obs, include_reward)
            self._update_cumulative_losses(cumulative_losses, current_losses=losses)
            batch_loss, batch_reward_loss, batch_terminal_loss, batch_latent_loss = self._extract_batch_loss(losses)
            progress_bar.set_postfix_str(f"loss={batch_loss} bce={batch_terminal_loss} gmm={batch_latent_loss} mse={batch_reward_loss}")
            progress_bar.update(self.batch_size)
        progress_bar.close()
        return cumulative_losses

    def _train_step(self, latent_obs, action, reward, terminal, latent_next_obs, include_reward):
        # reward = self._normalize_rewards(reward)
        losses,  batch_results = self._get_loss(latent_obs, action, reward, terminal, latent_next_obs, include_reward)
        self.optimizer.zero_grad()
        losses['loss'].backward()
        torch.nn.utils.clip_grad_norm_(parameters=self.mdrnn.parameters(),
                                       max_norm=self.config["mdrnn_trainer"]["gradient_clip"])
        self.optimizer.step()
        return losses,  batch_results

    def _test_step(self, latent_obs, action, reward, terminal, latent_next_obs, include_reward):
        # reward = self._normalize_rewards(reward)
        with torch.no_grad():
            return self._get_loss(latent_obs, action, reward, terminal, latent_next_obs, include_reward)

    def _test_batch(self, loader, include_reward):
        self.mdrnn.eval()
        try:
            batch = next(loader)

        except StopIteration:
            loader = iter(self.test_loader)
            batch = next(loader)

        latent_obs, actions, rewards, terminals, latent_next_obs = self._extract_batch_data(batch)
        losses, batch_results = self._test_step(latent_obs, actions, rewards, terminals, latent_next_obs, include_reward)

        batch_loss, batch_reward_loss, batch_terminal_loss, batch_latent_loss = self._extract_batch_loss(losses)
        self._log_batch_loss(batch_loss, batch_reward_loss, batch_terminal_loss, batch_latent_loss, self.baseline_test_loss, is_train=False)

        if self._loss_is_above_threshold(losses):
            print(f'Loss above threshold {losses}')

        if self._is_log_sample(losses):
            self._log_batch_prediction_results(batch_results, is_train=False)

        self.mdrnn.train()
        return loader

    def _get_loss(self, latent_obs, action, reward, terminal, latent_next_obs, include_reward: bool):
        """ Compute losses.
           The loss that is computed is:
           loss = (GMMLoss(latent_next_obs, GMMPredicted) + MSE(reward, predicted_reward) + BCE(terminal, logit_terminal)) / (LSIZE + 2)
           The LSIZE + 2 factor is here to counteract the fact that the GMMLoss scales
           approximately linearily with LSIZE. All losses are averaged both on the
           batch and the sequence dimensions (the two first dimensions).
           """
        latent_obs, action, reward, terminal, latent_next_obs = [arr.transpose(1, 0)
                                                                for arr in
                                                                [latent_obs, action, reward, terminal, latent_next_obs]]
        mus, sigmas, logpi, rs, ds, _ = self.mdrnn(action, latent_obs)
        batch_results = {
            'pred_gmm': {
                'mus': mus,
                'sigmas': sigmas,
                'logpi': logpi,
            },
            'pred_rewards': rs,
            'pred_terminals': ds,
            'target_rewards': reward,
            'target_terminals': terminal,
            'target_latents': latent_next_obs,
            'input_latent': latent_obs,
            'input_actions': action
        }

        gmm = gmm_loss(latent_next_obs, mus, sigmas, logpi)
        bce = f.binary_cross_entropy_with_logits(ds, terminal)
        if include_reward:
            mse = f.mse_loss(rs, reward)
            scale = self.latent_size + 2
        else:
            mse = 0
            scale = self.latent_size + 1
        loss = (gmm + bce + mse) / scale

        return dict(gmm=gmm, bce=bce, mse=mse, loss=loss), batch_results

    def _extract_batch_data(self, batch):
        obs, actions, rewards, terminals, next_obs = [arr.to(self.device) for arr in batch]
        latent_obs, latent_next_obs = self._to_latent(obs, next_obs)
        return latent_obs, actions, rewards, terminals, latent_next_obs

    def _to_latent(self, obs, next_obs):
        """ Transform observations to latent space.  """
        image_height, image_width = self.config['preprocessor']['img_height'], self.config['preprocessor']['img_width']
        latent_output_size = self.config['latent_size']

        with torch.no_grad():
            obs, next_obs = [
                f.interpolate(x.view(-1, self.config['preprocessor']['num_channels'], image_height, image_width),
                              size=latent_output_size, mode='bilinear', align_corners=True)
                for x in (obs, next_obs)]

            (obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma) = [self.vae(x)[1:] for x in (obs, next_obs)]

            latent_obs, latent_next_obs = [(x_mu + x_logsigma.exp() * torch.randn_like(x_mu)).view(self.batch_size,
                                                                                                   self.sequence_length,
                                                                                                   self.latent_size)
                                           for x_mu, x_logsigma in
                                           [(obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma)]]
        return latent_obs, latent_next_obs

    def _normalize_rewards(self, rewards):
        for batch in rewards:
            for i, reward in enumerate(batch):
                batch[i] = -1 if reward == -100 else reward
        return rewards

    def _update_cumulative_losses(self, cumulative_losses, current_losses):
        cumulative_losses['loss'] += current_losses['loss'].item()
        cumulative_losses['latent_loss'] += current_losses['gmm'].item()
        cumulative_losses['terminal_loss'] += current_losses['bce'].item()
        cumulative_losses['reward_loss'] += current_losses['mse'].item() if hasattr(current_losses['mse'], 'item') \
                                                                         else current_losses['mse']
        return cumulative_losses

    def _is_batch_testing(self):
        is_initial = self.batch_train_idx == 0
        is_batch_idx_equals_test = self.batch_train_idx % self.N_train_batch_until_test_batch == 0
        return is_initial or is_batch_idx_equals_test

    def _is_log_sample(self, losses=None):
        is_initial = self.batch_train_idx == 0
        is_batch_idx_equals_sampling = self.batch_train_idx % self.N_train_batch_until_pred_sampling == 0
        is_loss_above_treshold = self._loss_is_above_threshold(losses) if losses else False

        return is_initial or is_batch_idx_equals_sampling or is_loss_above_treshold

    def _loss_is_above_threshold(self, losses):
        loss_alert_threshold = 2  # general loss seems to be below 2 so anything above is suspecious
        for loss_key in losses:
            loss = losses[loss_key] / self.latent_size if loss_key == 'gmm' else losses[loss_key]
            if loss > loss_alert_threshold:
                return True
        return False

    def _extract_epoch_loss(self, cumulative_losses, loader):
        loss = cumulative_losses['loss'] * self.batch_size / len(loader.dataset)
        latent_loss = cumulative_losses['latent_loss'] / self.config['latent_size'] * self.batch_size / len(loader.dataset)
        terminal_loss = cumulative_losses['terminal_loss'] * self.batch_size / len(loader.dataset)
        reward_loss = cumulative_losses['reward_loss'] * self.batch_size / len(loader.dataset)
        return loss, latent_loss, terminal_loss, reward_loss

    def _extract_batch_loss(self, losses):
        batch_loss = losses['loss']
        batch_reward_loss = losses['mse']
        batch_terminal_loss = losses['bce']
        batch_latent_loss = losses['gmm'] / self.config['latent_size']
        return batch_loss, batch_reward_loss, batch_terminal_loss, batch_latent_loss

    def _log_batch_loss(self, loss, mse, bce, gmm, baseline_loss, is_train):
        self.logger.log_average_loss_per_batch(f'mdrnn', loss, self.batch_train_idx, is_train=is_train)
        self.logger.log_reward_loss_per_batch(f'mdrnn', mse, self.batch_train_idx, is_train=is_train)
        self.logger.log_terminal_loss_per_batch(f'mdrnn', bce, self.batch_train_idx, is_train=is_train)
        self.logger.log_next_latent_loss_per_batch(f'mdrnn', gmm, self.batch_train_idx, is_train=is_train)

        if self.is_baseline_reward_loss:
            self.logger.log_baseline_reward_loss_per_batch(f'mdrnn', baseline_loss, self.batch_train_idx, is_train=is_train)

    def _log_epoch_loss(self, average_loss, reward_loss, terminal_loss, next_latent_loss, baseline_loss, epoch, is_train):
        self.logger.log_average_loss_per_epoch('mdrnn', average_loss, epoch, is_train=is_train)
        self.logger.log_reward_loss_per_epoch('mdrnn', reward_loss, epoch, is_train=is_train)
        self.logger.log_terminal_loss_per_epoch('mdrnn', terminal_loss, epoch, is_train=is_train)
        self.logger.log_next_latent_loss_per_epoch('mdrnn', next_latent_loss, epoch, is_train=is_train)

        if self.is_baseline_reward_loss:
            self.logger.log_baseline_reward_loss_per_epoch('mdrnn', baseline_loss, epoch, is_train=is_train)

    def _log_batch_prediction_results(self, batch_results, is_train=True):
        predicted_gmm = batch_results['pred_gmm']
        pred_rewards = batch_results['pred_rewards']
        pred_terminals = batch_results['pred_terminals']
        target_rewards = batch_results['target_rewards']
        target_terminals = batch_results['target_terminals']
        target_latents = batch_results['target_latents']
        input_latent = batch_results['input_latent']
        input_actions = batch_results['input_actions']

        input_action_samples = self._get_init_mid_end_batch_results(input_actions)
        input_latent_samples = self._get_init_mid_end_batch_results(input_latent)
        input_decoded_frames_samples = [self._decode_latent_z(latent.unsqueeze(0)) for latent in input_latent_samples]

        pred_reward_samples = self._get_init_mid_end_batch_results(pred_rewards)
        pred_terminal_samples = self._get_init_mid_end_batch_results(pred_terminals)
        pred_latent_z_samples = self._get_pred_latents_from_batch(predicted_gmm)
        pred_decoded_frames_samples = [self._decode_latent_z(latent) for latent in pred_latent_z_samples]

        target_reward_samples = self._get_init_mid_end_batch_results(target_rewards)
        target_terminal_samples = self._get_init_mid_end_batch_results(target_terminals)
        target_latent_samples = self._get_init_mid_end_batch_results(target_latents)
        target_decoded_frames_samples = [self._decode_latent_z(latent.unsqueeze(0)) for latent in target_latent_samples]

        batch_result_samples = {
            'input_actions': [action.clone().detach().numpy() for action in input_action_samples],
            'input_frames': input_decoded_frames_samples,
            'target_rewards': [reward.clone().detach().numpy() for reward in target_reward_samples],
            'target_terminals': [terminal.clone().detach().numpy() for terminal in target_terminal_samples],
            'target_frames': target_decoded_frames_samples,
            'pred_rewards': [reward.clone().detach().numpy() for reward in pred_reward_samples],
            'pred_terminals': [terminal.clone().detach().numpy() for terminal in pred_terminal_samples],
            'pred_frames': pred_decoded_frames_samples
        }

        self.logger.log_batch_sample(samples=batch_result_samples, batch_idx=self.batch_train_idx, is_train=is_train)

    def _calc_reward_avg_baseline(self):
        with torch.no_grad():
            loader = self.train_loader
            num_rewards, reward_sum = torch.tensor(0, dtype=torch.int32), torch.tensor(0.0, dtype=torch.float32)
            progress_bar = tqdm(total=len(loader.dataset) if len(loader.dataset) >= 0 else 1, desc=f'Calculating avg reward_baseline')
            for i, batch in enumerate(loader):
                _, _, reward_rollouts, _, _ = batch
                reward_sum += torch.sum(reward_rollouts)
                num_rewards += reward_rollouts.shape[1]
                progress_bar.update(self.batch_size)
    
            avg_baseline_reward = reward_sum.item()/num_rewards.item()
            self.logger.log_reward_baseline_value('mdrnn', self.session_name, avg_baseline_reward, num_rewards)
            print(f'Avg baseline reward: {avg_baseline_reward} | number of rewards: {num_rewards}')
            return avg_baseline_reward

    def _calc_baseline_losses(self, reward_avg_baseline):
        with torch.no_grad():
            train_loss = self._calc_baseline_loss(self.train_loader, reward_avg_baseline, 'train')
            test_loss = self._calc_baseline_loss(self.test_loader, reward_avg_baseline, 'test')
            self.logger.log_reward_baseline_losses('mdrnn',  self.session_name, train_loss, test_loss)
            print(f'baseline train loss: {train_loss} | baseline test loss: {test_loss}')
            return train_loss, test_loss

    def _calc_baseline_loss(self, loader, reward_avg_baseline, tag=''):
        progress_bar = tqdm(total=len(loader.dataset) if len(loader.dataset) >= 0 else 1, desc=f'Calculating reward baseline loss {tag}')
        rewards = []
        for i, batch in enumerate(loader):
            _, _, reward_rollouts, _, _ = batch
            rewards.append(reward_rollouts.numpy())
            progress_bar.update(self.batch_size)
        progress_bar.close()
        return f.mse_loss(torch.tensor(rewards), torch.tensor(reward_avg_baseline)).item()

    def _get_init_mid_end_batch_results(self, arr):
        initial = arr[0][0].cpu()
        mid = arr[int(self.sequence_length / 2)][0].cpu()
        end = arr[self.sequence_length - 1][0].cpu()
        return [initial, mid, end]

    def _get_pred_latents_from_batch(self, predicted_gmm):
        pred_means_samples = self._get_init_mid_end_batch_results(predicted_gmm['mus'])
        pred_means_samples = [mean.unsqueeze(0).unsqueeze(0) for mean in pred_means_samples]

        pred_sigma_samples = self._get_init_mid_end_batch_results(predicted_gmm['sigmas'])
        pred_sigma_samples = [sigma.unsqueeze(0).unsqueeze(0) for sigma in pred_sigma_samples]

        pred_logpi_samples = self._get_init_mid_end_batch_results(predicted_gmm['logpi'])
        pred_logpi_samples = [logpi.unsqueeze(0).unsqueeze(0) for logpi in pred_logpi_samples]

        pred_latent_z_samples = [self._sample_next_z(mean, sigma, logpi) for mean in pred_means_samples
                                                                         for sigma in pred_sigma_samples
                                                                         for logpi in pred_logpi_samples]

        return pred_latent_z_samples

    def _sample_next_z(self, z_means, z_standard_deviations, log_mixture_weights):  # input: (1, 1, 5, 32) --> (seq_len, batch_size, num_gaussians, latent_size)
        temperature = 1.15   # https://worldmodels.github.io/
        log_mixture_weights = self._adjust_mixture_weights_by_temperature(log_mixture_weights, temperature)
        random_gaussian_mixture_index = Categorical(log_mixture_weights).sample().item()
        sampled_mean = z_means[:, :, random_gaussian_mixture_index, :]
        sampled_standard_deviation = z_standard_deviations[:, :, random_gaussian_mixture_index, :]
        random_gaussian_noise = torch.randn_like(z_means[:, :, random_gaussian_mixture_index, :])  #* torch.sqrt(torch.as_tensor(temperature))
        random_gaussian_noise = random_gaussian_noise * torch.sqrt(torch.as_tensor(temperature))
        next_latent_z = sampled_mean + sampled_standard_deviation * random_gaussian_noise
        return next_latent_z.squeeze(0)  # (1, 1, 32) --> (1, 32)

    def _adjust_mixture_weights_by_temperature(self, log_mixture_weights, temperature):
        # Paper: https://arxiv.org/pdf/1704.03477.pdf
        # Code: https://github.com/tensorflow/magenta/blob/master/magenta/models/sketch_rnn/model.py
        log_mixture_weights /= temperature
        log_mixture_weights -= log_mixture_weights.max()
        log_mixture_weights = torch.exp(log_mixture_weights)
        log_mixture_weights /= log_mixture_weights.sum()  # Softmax normalize
        return log_mixture_weights

    def _decode_latent_z(self, latent_z):
        with torch.no_grad():
            latent_z = latent_z.to(self.device)
            reconstructed_frame = self.vae.decoder(latent_z)
            frame = reconstructed_frame.cpu().numpy()
            frame = np.clip(frame, 0, 1) * 255
            frame = np.transpose(frame, (0, 2, 3, 1))
            frame = frame.squeeze()
            frame = frame.astype(np.uint8)
            return frame