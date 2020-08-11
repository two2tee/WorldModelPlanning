""" VAE implementation """
#  Copyright (c) 2020, - All Rights Reserved
#  This file is part of the Evolutionary Planning on a Learned World Model thesis.
#  Unauthorized copying of this file, via any medium is strictly prohibited without the consensus of the authors.
#  Written by Thor V.A.N. Olesen <thorolesen@gmail.com> & Dennis T.T. Nguyen <dennisnguyen3000@yahoo.dk>.

from __future__ import print_function  # NB: must be beginning of file
from torch import nn
from torch.nn import functional as F
import torch


class Encoder(nn.Module):
    def __init__(self, config, latent_size):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=config['convolution']['layer_1']['in'],
                               out_channels=config['convolution']['layer_1']['out'],
                               kernel_size=config['convolution']['layer_1']['filter_size'],
                               stride=config['convolution']['layer_1']['strides'])

        self.conv2 = nn.Conv2d(config['convolution']['layer_2']['in'],
                               config['convolution']['layer_2']['out'],
                               config['convolution']['layer_2']['filter_size'],
                               config['convolution']['layer_2']['strides'])

        self.conv3 = nn.Conv2d(config['convolution']['layer_3']['in'],
                               config['convolution']['layer_3']['out'],
                               config['convolution']['layer_3']['filter_size'],
                               config['convolution']['layer_3']['strides'])

        self.conv4 = nn.Conv2d(config['convolution']['layer_4']['in'],
                               config['convolution']['layer_4']['out'],
                               config['convolution']['layer_4']['filter_size'],
                               config['convolution']['layer_4']['strides'])

        self.z_fully_connected_mean = nn.Linear(in_features=config['dense']['in'],
                                                out_features=latent_size)

        self.z_fully_connected_log_standard_deviation = nn.Linear(in_features=config['dense']['in'],
                                                                  out_features=latent_size)

    def flatten(self, x):
        return x.view(x.size(0), -1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.flatten(x)
        z_mean = self.z_fully_connected_mean(x)
        z_log_standard_deviation = self.z_fully_connected_log_standard_deviation(x)  # Is log variance not std - breaks existing models if changed
        return z_mean, z_log_standard_deviation


class Decoder(nn.Module):
    def __init__(self,  config, latent_size):
        super(Decoder, self).__init__()

        self.fully_connected = nn.Linear(latent_size, config['dense']['out'])

        self.deconv1 = nn.ConvTranspose2d(in_channels=config['deconvolution']['layer_1']['in'],
                                          out_channels=config['deconvolution']['layer_1']['out'],
                                          kernel_size=config['deconvolution']['layer_1']['filter_size'],
                                          stride=config['deconvolution']['layer_1']['strides'])

        self.deconv2 = nn.ConvTranspose2d(config['deconvolution']['layer_2']['in'],
                                          config['deconvolution']['layer_2']['out'],
                                          config['deconvolution']['layer_2']['filter_size'],
                                          config['deconvolution']['layer_2']['strides'])

        self.deconv3 = nn.ConvTranspose2d(config['deconvolution']['layer_3']['in'],
                                          config['deconvolution']['layer_3']['out'],
                                          config['deconvolution']['layer_3']['filter_size'],
                                          config['deconvolution']['layer_3']['strides'])

        self.deconv4 = nn.ConvTranspose2d(config['deconvolution']['layer_4']['in'],
                                          config['deconvolution']['layer_4']['out'],
                                          config['deconvolution']['layer_4']['filter_size'],
                                          config['deconvolution']['layer_4']['strides'])

    def unflatten(self, x):
        return x.unsqueeze(-1).unsqueeze(-1)

    def forward(self, x):
        x = F.relu(self.fully_connected(x))
        x = self.unflatten(x)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        reconstruction = torch.sigmoid(self.deconv4(x))
        return reconstruction


class VAE(nn.Module):
    def __init__(self, config):
        super(VAE, self).__init__()
        self.encoder = Encoder(config['vae']['encoder'], config['latent_size'])
        self.decoder = Decoder(config['vae']['decoder'], config['latent_size'])

    def sample_reparametarization(self, mean, log_variance):
        log_variance = log_variance.exp()
        gaussian_noise = torch.randn_like(log_variance)  # epsilon ~ N(0,I) => warnings: non-determinism in model
        z_latent = mean + log_variance * gaussian_noise
        return z_latent  # Sample Gaussian prior z ~ N(0, σI) = mu + σ * N(0, I)

    def forward(self, x):
        z_mean, z_log_variance = self.encoder(x)
        z_latent = self.sample_reparametarization(z_mean, z_log_variance)
        reconstruction = self.decoder(z_latent)
        return reconstruction, z_mean, z_log_variance
