""" MDRNN Implementation - MDRNN is a conjunction of an MDN and LSTM """
#  Copyright (c) 2020, - All Rights Reserved
#  This file is part of the Evolutionary Planning on a Learned World Model thesis.
#  Unauthorized copying of this file, via any medium is strictly prohibited without the consensus of the authors.
#  Written by Thor V.A.N. Olesen <thorolesen@gmail.com> & Dennis T.T. Nguyen <dennisnguyen3000@yahoo.dk>.

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, num_actions, latent_size=32, num_hidden_units=256):
        super(LSTM, self).__init__()
        self.input_size = latent_size + num_actions
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=num_hidden_units)

    def forward(self, latents_and_actions_input, previous_hidden_state=None):
        output, (next_hidden_state, next_cell_state) = self.lstm(latents_and_actions_input, previous_hidden_state)
        return output, (next_hidden_state, next_cell_state)


class MDN(nn.Module):  # Mixture Density Network
    def __init__(self, latent_size=32, num_gaussians=5, num_hidden_units=256):
        super(MDN, self).__init__()
        self.pi_weight_size = 1
        self.means_size = latent_size
        self.standard_deviations_size = latent_size
        self.reward_done_size = 2  # TODO:DEV
        self.output_size = num_gaussians * (self.means_size + self.standard_deviations_size + self.pi_weight_size) + self.reward_done_size
        self.linear_gaussian_mixture_model = nn.Linear(in_features=num_hidden_units, out_features=self.output_size)

    def forward(self, hidden_units):
        return self.linear_gaussian_mixture_model(hidden_units)


class MDRNN(nn.Module):
    def __init__(self, latent_size, num_gaussians, num_hidden_units, num_actions):
        super(MDRNN, self).__init__()
        self.latent_size = latent_size
        self.num_gaussians = num_gaussians
        self.stride = self.num_gaussians * self.latent_size
        self.lstm = LSTM(num_actions, latent_size, num_hidden_units)
        self.mdn = MDN(latent_size, num_gaussians, num_hidden_units)

    def forward(self, actions, latents, previous_hidden_states=None):  # batch of (seq=#actions, batch=#sequences) -> action in R3
        sequence_length, batch_size = actions.size(0), actions.size(1)  # rows 0 , columns 1
        actions_latents_inputs = torch.cat([actions, latents], dim=-1)  # auto infer dims --> (z: 32, a: 3 -> 35)

        lstm_outputs, (next_hidden_states, next_cell_states) = self.lstm(actions_latents_inputs, previous_hidden_states)
        mdn_outputs = self.mdn(lstm_outputs)

        means = mdn_outputs[:, :, :self.stride].view(sequence_length, batch_size, self.num_gaussians, self.latent_size)  # mu

        log_standard_deviations = mdn_outputs[:, :, self.stride:2 * self.stride].view(sequence_length, batch_size, self.num_gaussians, self.latent_size)  # (seq, batch, input) -> (1,1,5,32)
        standard_deviations = torch.exp(log_standard_deviations)  # sigma

        mixture_weights = mdn_outputs[:, :, 2 * self.stride: 2 * self.stride + self.num_gaussians]  # pi (seq, batch, input) -> (1,1,5,32)
        log_mixture_weights = F.log_softmax(mixture_weights, dim=-1)  # normalization of mixture weights, infer dims

        rewards = mdn_outputs[:, :, -2]  # TODO:DEV
        dones = mdn_outputs[:, :, -1]

        return means, standard_deviations, log_mixture_weights, rewards, dones, (next_hidden_states, next_cell_states)
