"""Simulated environment to enable play and planning in world model"""
#  Copyright (c) 2020, - All Rights Reserved
#  This file is part of the Evolutionary Planning on a Learned World Model thesis.
#  Unauthorized copying of this file, via any medium is strictly prohibited without the consensus of the authors.
#  Written by Thor V.A.N. Olesen <thorolesen@gmail.com> & Dennis T.T. Nguyen <dennisnguyen3000@yahoo.dk>.

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import math
from torch.distributions.categorical import Categorical
matplotlib.use('Qt5Agg')  # Required for Python, Matplotlib 3 on Mac OSX


class SimulatedEnvironment:
    def __init__(self, config, vae, mdrnn):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.mdrnn = mdrnn.cpu()
        self.vae = vae.cpu()
        self.is_discretize_sampling = config['planning']['is_discretize_sampling']
        self.temperature = config['simulated_environment']['temperature']
        self.is_render_reconstructions = config['visualization']['is_render_reconstructions']

        # mdrnn states
        self.current_latent_state_z = None
        self.current_hidden_states = None
        self._reset_states()

        # rendering
        if self.is_render_reconstructions or config['is_play'] or config['test_suite']['is_run_model_tests'] \
                                          or config['visualization']['is_render_simulation'] or config['visualization']['is_render_dream']:
            self.monitor = None
            self.figure = plt.figure(num=random.randint(0, 999))
            self.figure_num = self.figure.number
            self.current_reconstruction = None

        # sampling
        self.steer_delta = self.config['simulated_environment']['steer_delta']
        self.gas_delta = self.config['simulated_environment']['gas_delta']
        self.max_gas = self.config['simulated_environment']['max_gas']
        self.max_brake = self.config['simulated_environment']['max_brake']

    def step(self, action, hidden_state_h=None, latent_state_z=None, is_simulation_real_environment=True):
        hidden_state_h = self.current_hidden_states if hidden_state_h is None else hidden_state_h
        latent_state_z = self.current_latent_state_z if latent_state_z is None else latent_state_z
        next_latent_state_z, rewards, dones, next_hidden_states = self._step_mdrnn(action, latent_state_z, hidden_state_h)

        if is_simulation_real_environment:  # Keep track of latent and hidden states if hallucination is real environment
            self.current_reconstruction = self._decode_latent_z(next_latent_state_z)  # Render
            self.current_hidden_states = next_hidden_states
            self.current_latent_state_z = next_latent_state_z

        # (z', r, d, h')
        return next_latent_state_z.clone().detach(), rewards.item(), (dones > 0).item(), [next_hidden_states[0].clone().detach(), next_hidden_states[1].clone().detach()]

    def reset(self):
        if self.is_render_reconstructions:
            self._reset_monitor()
        return self._reset_states()

    def _step_mdrnn(self, action, latent_z, hidden_states):
        with torch.no_grad():
            action = torch.Tensor(action).unsqueeze(0).unsqueeze(0)
            latent_z = latent_z.unsqueeze(0)
            means, standard_deviations, log_mixture_weights, rewards, dones, next_hidden_states = self.mdrnn.forward(action, latent_z, hidden_states)
            log_mixture_weights = log_mixture_weights.squeeze()
            next_latent_z = self._sample_next_z(means, standard_deviations, log_mixture_weights)
            return next_latent_z, rewards, dones, next_hidden_states

    def _sample_next_z(self, z_means, z_standard_deviations, log_mixture_weights):  # input: (1, 1, 5, 32) --> (seq_len, batch_size, num_gaussians, latent_size)
        # Inspiration: https://github.com/hardmaru/WorldModelsExperiments/blob/244f79c2aaddd6ef994d155cd36b34b6d907dcfe/carracing/dream_env.py#L70
        log_mixture_weights = log_mixture_weights if self.temperature <= 0 else self._adjust_mixture_weights_by_temperature(log_mixture_weights, self.temperature)
        random_gaussian_mixture_index = Categorical(log_mixture_weights).sample().item()
        sampled_mean = z_means[:, :, random_gaussian_mixture_index, :]
        sampled_standard_deviation = z_standard_deviations[:, :, random_gaussian_mixture_index, :]
        random_gaussian_noise = torch.randn_like(z_means[:, :, random_gaussian_mixture_index, :])  #* torch.sqrt(torch.as_tensor(self.temperature))
        random_gaussian_noise = random_gaussian_noise if self.temperature <= 0 else random_gaussian_noise * torch.sqrt(torch.as_tensor(self.temperature))

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
            reconstructed_frame = self.vae.decoder(latent_z.cpu())
            frame = reconstructed_frame.numpy()
            frame = np.clip(frame, 0, 1) * 255
            frame = np.transpose(frame, (0, 2, 3, 1))
            frame = frame.squeeze()
            frame = frame.astype(np.uint8)
            return frame

    def _reset_states(self):
        self.current_latent_state_z = torch.randn(1, self.config['latent_size'])  # Random latent z
        self.current_hidden_states = self.get_hidden_zeros_state()  # LSTM Hidden state reset
        return self._decode_latent_z(self.current_latent_state_z)

    def _reset_monitor(self):
        if not self.monitor:
            plt.figure(num=self.figure_num)
            plt.clf()
            self.monitor = plt.imshow(X=np.zeros((self.config['preprocessor']['img_width'],
                                                  self.config['preprocessor']['img_height'],
                                                  self.config['preprocessor']['num_channels']),
                                      dtype=np.uint8))

    def render(self, reconstruction=None):
        if not self.monitor:
            self._reset_monitor()
        plt.figure(num=self.figure_num)
        plt.title('VAE Reconstruction') if reconstruction is not None else plt.title('MDRNN Reconstruction')
        reconstruction = self.current_reconstruction if reconstruction is None else reconstruction.squeeze().permute(1, 2, 0)  # align image dimensions
        self.monitor.set_data(reconstruction)
        plt.pause(.01)

    def sample(self, previous_action=None):  # Sampling: [ steer, gas, brake ] = [ [-1, +1] , [0, 1], [0, 1] ]
        return self._continous_sample(previous_action) if not self.is_discretize_sampling else self.discrete_sample()

    def get_hidden_zeros_state(self):
        return 2 * [torch.zeros(1, self.config['mdrnn']['hidden_units']).unsqueeze(0)]

    def _continous_sample(self, previous_action=None):
        steer = np.random.uniform(low=-1, high=1) if previous_action is None else \
                np.random.choice([max(previous_action[0] - self.steer_delta, -1),
                                  previous_action[0],
                                  min(previous_action[0] + self.steer_delta, 1)])

        gas = np.random.uniform(low=self.max_brake, high=self.max_gas) if previous_action is None else \
              np.random.choice([max(previous_action[1] - self.gas_delta, -1),
                                previous_action[1],
                                min(previous_action[1] + self.gas_delta, 1)])

        # Brake: negative sign of gas to avoid simultaneous brake/gas driving
        gas = gas if gas > 0 else 0
        brake = abs(gas) if gas < 0 else 0
        return [steer, gas, brake]

    def discrete_sample(self):
        steer_steps = [round(e, 1) for e in np.arange(start=-1.0, stop=1.0, step=0.1)]
        gas_steps = [round(e, 1) for e in np.arange(start=-1.0, stop=1.0, step=0.2)]
        steer, gas = np.random.choice(steer_steps), np.random.choice(gas_steps)
        gas = gas if gas > 0 else 0
        brake = abs(gas) if gas < 0 else 0
        return [steer, gas, brake]

    def brownian_sample(self, previous_action, delta):  # a_{t+1} = a_t + sqrt(dt) N(0, 1)
        dactions_dt = np.random.randn(len(previous_action))
        new_action = [0, 0, 0]
        new_action[0] = np.clip(previous_action[0] + math.sqrt(delta) * dactions_dt[0], -1, 1)
        new_action[1] = np.clip(previous_action[1] + math.sqrt(delta) * dactions_dt[1], 0, 1)
        new_action[2] = np.clip(previous_action[2] + math.sqrt(delta) * dactions_dt[2], 0, 1)
        return new_action

    def discrete_action_space(self, action=None):
        actions = set()
        steer_steps = np.arange(start=-1.0, stop=1.0, step=self.steer_delta) if action is None else [max(action[0] - self.steer_delta, -1), action[0], min(action[0] + self.steer_delta, 1)]
        gas_steps = np.arange(start=-1.0, stop=1.0, step=self.gas_delta) if action is None else [max(action[1] - self.gas_delta, -1), action[1], min(action[1] + self.gas_delta, 1)]
        steer_steps, gas_steps = [round(e, 1) for e in steer_steps], [round(e, 1) for e in gas_steps]  # Remove decimal precision

        for steer in steer_steps:
            for gas in gas_steps:
                actions.add((steer, gas, 0)) if gas > 0 else actions.add((steer, 0, abs(gas)))  # negative sign gas = brake

        return [list(a) for a in actions]

    def discrete_delta_sample(self, previous_action=None):
        actions = self.discrete_action_space(previous_action)
        random_index = random.randrange(len(actions))
        return actions[random_index]
