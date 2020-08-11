#  Copyright (c) 2020, - All Rights Reserved
#  This file is part of the Evolutionary Planning on a Learned World Model thesis.
#  Unauthorized copying of this file, via any medium is strictly prohibited without the consensus of the authors.
#  Written by Thor V.A.N. Olesen <thorolesen@gmail.com> & Dennis T.T. Nguyen <dennisnguyen3000@yahoo.dk>.
import torch

from utility.preprocessor import Preprocessor
from environment.simulated_environment import SimulatedEnvironment


class AgentWrapper:
    def __init__(self, agent, config, vae, mdrnn):
        self.vae = vae
        self.mdrnn = mdrnn
        self.agent = agent
        self.latent = None
        self.config = config
        self.preprocessor = Preprocessor(config['preprocessor'])
        self.simulated_environment = SimulatedEnvironment(config, vae, mdrnn)
        self.hidden = self.simulated_environment.get_hidden_zeros_state()

    def search(self, state):
        self.latent = self._compress(state)
        action, _ = self.agent.search(self.simulated_environment, self.latent, self.hidden)
        return action

    def synchronize(self, next_state, action):
        latent_state = self._compress(next_state)
        self.latent, _, _, self.hidden = self.simulated_environment.step(action, self.hidden, latent_state,
                                                                         is_simulation_real_environment=True)

    def reset(self):
        self.latent = None
        self.hidden = self.simulated_environment.get_hidden_zeros_state()
        self.simulated_environment.reset()

    def _compress(self, state):
        with torch.no_grad():
            state = self.preprocessor.resize_frame(state).unsqueeze(0)
            _, z_mean, z_log_standard_deviation = self.vae(state)
            latent_state = self.vae.sample_reparametarization(z_mean, z_log_standard_deviation)
            return latent_state
