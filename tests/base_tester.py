import multiprocessing
from environment.simulated_environment import SimulatedEnvironment

multiprocessing.set_start_method('spawn') if multiprocessing.get_start_method() is None else None

class BaseTester:
    def __init__(self, config, vae, mdrnn, preprocessor, environment, trials):
        self.config = config
        self.environment = environment
        self.mdrnn = mdrnn
        self.vae = vae
        self.seed = None
        self.preprocessor = preprocessor
        self.simulated_environment = SimulatedEnvironment(self.config, self.vae, self.mdrnn)
        self.trials = trials

    def _encode_state(self, state):
        state = self.preprocessor.resize_frame(state).unsqueeze(0)
        decoded_state, z_mean, z_log_standard_deviation = self.vae(state)
        latent_state = self.vae.sample_reparametarization(z_mean, z_log_standard_deviation)
        return latent_state, decoded_state
