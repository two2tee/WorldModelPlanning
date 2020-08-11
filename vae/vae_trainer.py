"""" VAE Trainer used to train or reload VAEs """
#  Copyright (c) 2020, - All Rights Reserved
#  This file is part of the Evolutionary Planning on a Learned World Model thesis.
#  Unauthorized copying of this file, via any medium is strictly prohibited without the consensus of the authors.
#  Written by Thor V.A.N. Olesen <thorolesen@gmail.com> & Dennis T.T. Nguyen <dennisnguyen3000@yahoo.dk>.

import sys
import torch
import multiprocessing
from os import mkdir
from os.path import exists, join
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utility.loaders import RolloutObservationDataset
sys.path.append("..")


class VaeTrainer:
    def __init__(self, config, preprocesser, logger):
        torch.backends.cudnn.benchmark = True
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vae = None
        self.logger = logger
        self.vae_name = self.config['forced_vae'] if self.config['forced_vae'] else self.config['experiment_name'].replace('iterative_', '')
        self.session_name = 'vae_'+self.config['experiment_name'].replace('iterative_', '')
        self.preprocessor = preprocesser
        self.model_dir = self.config['vae_dir']
        self.data_dir = self.config['data_dir']
        self.best_vae_filename = "checkpoints/" + self.vae_name+'_'+self.config["vae_trainer"]["vae_best_filename"]
        self.checkpoint_filename = "checkpoints/" + self.vae_name+'_'+self.config["vae_trainer"]["vae_checkpoint_filename"]
        self.is_save_reconstruction = self.config['vae_trainer']['is_save_reconstruction']
        self.batch_size = self.config['vae_trainer']['batch_size']
        self.train_buffer_size = self.config['vae_trainer']['train_buffer_size']
        self.test_buffer_size = self.config['vae_trainer']['test_buffer_size']

        if not exists(self.model_dir):
            mkdir(self.model_dir)

        self.num_workers = self.config['vae_trainer']['num_workers']
        self.num_workers = self.num_workers if self.num_workers <= multiprocessing.cpu_count() else multiprocessing.cpu_count()

        self.train_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.test_loader = None

        self.optimizer = None

    def load_data(self):  # To avoid reloading in constructor when not training
        self.train_dataset = RolloutObservationDataset(self.data_dir, self.preprocessor.normalize_frames_train, is_train=True, buffer_size=self.train_buffer_size)
        self.test_dataset = RolloutObservationDataset(self.data_dir, self.preprocessor.normalize_frames_test, is_train=False, buffer_size=self.test_buffer_size)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def train(self, vae):
        self.load_data()
        self.vae = vae
        self.logger.is_logging = True
        self.logger.start_log_training_minimal(name=self.session_name)
        self.vae = self.vae.to(self.device)
        self.optimizer = optim.Adam(self.vae.parameters(), lr=self.config['vae_trainer']['learning_rate'])
        start_epoch, current_best = self._reload_training_session()
        max_epochs = self.config['vae_trainer']['max_epochs']

        if start_epoch > max_epochs:
            raise Exception(f'Inconsistent start epoch {start_epoch} and max_epoch {max_epochs}')

        for epoch in range(start_epoch, max_epochs + 1):
            self._train_epoch(epoch)
            test_loss = self._test_epoch(epoch)
            is_best = not current_best or test_loss < current_best
            current_best = test_loss if is_best else current_best

            self._save_checkpoint({'epoch': epoch, 'state_dict': self.vae.state_dict(),
                                   'precision': test_loss, 'optimizer': self.optimizer.state_dict(),
                                   }, is_best)
            if self.is_save_reconstruction:
                self._save_reconstructed_sample_frames(epoch)

        self.logger.end_log_training('vae')
        return self.vae

    def _save_reconstructed_sample_frames(self, epoch):
        with torch.no_grad():
            shape = (self.config['num_reconstructions'],  # 64 reconstructions
                     self.config['preprocessor']['num_channels'],  # 3 image channels
                     self.preprocessor.img_height, self.preprocessor.img_width)  # size 64x64
            z_samples = torch.randn(self.config['num_reconstructions'], self.config['latent_size']).to(self.device)
            reconstructed_sample_frames = self.vae.decoder(z_samples).cpu()  # Extract reconstruction from GPU
            reconstructed_sample_frames = reconstructed_sample_frames.view(shape)
            self.logger.log_vae_reconstruction(reconstructed_sample_frames, epoch)
        print(f'Reconstruction_{epoch} saved')

    def reload_model(self, vae, device=None):
        reload_file = join(self.model_dir, self.best_vae_filename)
        if not exists(reload_file):
            raise Exception('No VAE model found...')
        state = torch.load(reload_file, map_location=device if device else self.device)
        vae.load_state_dict(state['state_dict'])
        print('Reloaded VAE model')
        return vae

    def _reload_training_session(self):
        reload_file = join(self.model_dir, self.checkpoint_filename)
        best_file = join(self.model_dir, self.best_vae_filename)

        reload_file = reload_file if exists(reload_file) else best_file

        if exists(reload_file) and self.config['vae_trainer']['is_continue_model']:
            state = torch.load(reload_file, map_location=self.device)
            best_test_loss = None
            if exists(best_file):
                best_state = torch.load(best_file, map_location=self.device)
                best_test_loss = best_state['precision']
                print(f"Reloading vae at epoch {state['epoch']}, with best test error {best_test_loss} at epoch {best_state['epoch']}")
            else:
                print(f"Reloading vae at epoch {state['epoch']}")

            self.vae.load_state_dict(state['state_dict'])
            self.optimizer.load_state_dict(state['optimizer'])
            return state['epoch'], best_test_loss
        print('No vae found. Skip reloading...')
        return 1, None

    # Reconstruction + KL divergence losses summed over all elements and batch
    # https://arxiv.org/abs/1312.6114 : 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    def loss_function(self, reconstructed_frame, target_frame, mean, log_variance, kl_tolerance=0.5):
        # Binary Cross Entropy can be used if image pixels are normaliged [0,1] if not L2 loss for reconstruction
        reconstruction_loss = F.mse_loss(reconstructed_frame, target_frame, reduction='sum')
        # Regularization term based on closed formula assuming P(Z) ~ N(mu, sigma*I)
        kl_divergence_loss = - kl_tolerance * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp())
        return reconstruction_loss + kl_divergence_loss

    def _train_epoch(self, epoch):
        self.vae.train()  # Turn on train mode
        self.train_dataset.load_next_buffer()
        train_loss = 0
        progress_bar = tqdm(total=len(self.train_loader.dataset), desc=f"Train Epoch {epoch}")
        for batch_index, target_frames_batch in enumerate(self.train_loader):
            target_frames_batch = target_frames_batch.to(self.device)
            self.optimizer.zero_grad()  # reset gradient
            reconstructed_frame, mean, log_variance = self.vae(target_frames_batch)
            loss = self.loss_function(reconstructed_frame, target_frames_batch, mean, log_variance)
            loss.backward()  # back propagation
            train_loss += loss.item()  # Extract loss from tensor
            self.optimizer.step()  # Gradient Descent step
            progress_bar.update(self.batch_size)

        progress_bar.close()

        if len(self.train_loader.dataset) <= 0:
            print(f"Train data size issue with size: {len(self.train_loader.dataset)}")
            return

        train_loss = train_loss / len(self.train_loader.dataset)
        self.logger.log_loss('vae', train_loss, epoch, is_train=True)
        print(f'====> Epoch: {epoch} Average train loss: {train_loss / len(self.train_loader.dataset)}')

    def _test_epoch(self, epoch):
        self.vae.eval()  # Turn on test mode
        self.test_dataset.load_next_buffer()
        progress_bar = tqdm(total=len(self.test_loader.dataset), desc=f"Test Epoch {epoch}")
        test_loss = 0
        with torch.no_grad():
            for target_frame in self.test_loader:
                target_frame = target_frame.to(self.device)
                reconstructed_frame, mean, log_variance = self.vae(target_frame)
                test_loss += self.loss_function(reconstructed_frame, target_frame, mean, log_variance).item()
                progress_bar.set_postfix_str(f"test loss={test_loss}")
                progress_bar.update(self.batch_size)

        progress_bar.close()

        if len(self.test_loader.dataset) <= 0:
            print(f"Test data size issue with size: {len(self.test_loader.dataset)}")
            return

        test_loss /= len(self.test_loader.dataset)   # Average test loss
        print(f'====> Epoch: {epoch} Average test loss: {test_loss}')
        self.logger.log_loss('vae', test_loss, epoch, is_train=False)
        return test_loss

    def _save_checkpoint(self, state, is_best):
        best_filename = join(self.model_dir, self.best_vae_filename)
        filename = join(self.model_dir, self.checkpoint_filename)
        torch.save(state, filename)
        if is_best:
            torch.save(state, best_filename)
            print(f'New best model found and saved')
