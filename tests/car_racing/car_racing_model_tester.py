import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
from tests.base_tester import BaseTester
from gym.envs.box2d.car_dynamics import Car
from environment.simulated_environment import SimulatedEnvironment


class ModelTester(BaseTester):
    def __init__(self, config, vae, mdrnn, preprocessor, environment):
        super().__init__(config, vae, mdrnn, preprocessor, environment, trials=config["test_suite"]["trials"])
        self.seed = 9214
        self.vae_render = SimulatedEnvironment(self.config, self.vae, self.mdrnn)

    def run_tests(self):
        print('--- RUNNING MODEL TESTS ---')
        print(f'seed: {self.seed} | trials per test: {self.trials}')
        print(f'VAE Parameters')
        print(self.config['vae'])
        print(f'MDRNN Parameters')
        with torch.no_grad():
            self._forward_drive_slow_test()
            self._forward_drive_medium_test()
            self._forward_drive_fast_test()
            self._drive_to_grass_test()
            self._stand_still_test()
            self._drive_right_turn_test()
            self._drive_pass_right_turn_test()
            self._drive_left_turn_test()
            self._drive_pass_left_turn_test()
            self._drive_s_turn_test()
            self._drive_pass_s_turn_test()
            self._drive_u_turn_test()
            self._drive_pass_u_turn_test()
        plt.close('all')

    def _stand_still_test(self):
        print(f'\n----- Stand still test ------')
        actions = [([0, 0, 0], 80)]
        self._execute_trial(actions)

    def _forward_drive_slow_test(self):
        print(f'\n----- Forward drive slow test ------')
        actions = [([0, 0.01, 0], 80)]
        self._execute_trial(actions, start_track=25)

    def _forward_drive_medium_test(self):
        print(f'\n----- Forward drive medium test ------')
        actions = [([0, 0.1, 0], 60), ([0, 0.1, 0], 20)]
        self._execute_trial(actions, start_track=25)

    def _forward_drive_fast_test(self):
        print(f'\n----- Forward drive fast test ------')
        actions = [([0, 0.1, 0], 60), ([0, 1, 0], 20)]
        self._execute_trial(actions, start_track=25)

    def _forward_oscilation(self):
        print(f'\n----- Forward drive fast oscilation test ------')
        actions = [([0, 0.1, 0], 10), ([0, 1, 0], 20)]
        self._execute_trial(actions, start_track=25)

    def _drive_to_grass_test(self):
        print(f'\n----- Drive to grass test ------')
        actions = [([1, 0.5, 0], 30), ([0, 0.3, 0], 50)]  # actions: steer right, drive and break
        self._execute_trial(actions, start_track=25)

    def _drive_left_turn_test(self):
        print(f'\n----- Drive left turn test ------')
        actions = [([0, 0.5, 0], 20), ([-0.3, 0.5, 0], 20), ([-0.2, 0.5, 0], 10), ([0, 0.1, 0], 30)]
        self._execute_trial(actions, start_track=14)

    def _drive_pass_left_turn_test(self):
        print(f'\n----- Drive pass left turn test ------')
        actions = [([0, 0.3, 0], 80)]
        self._execute_trial(actions, start_track=14)

    def _drive_right_turn_test(self):
        print(f'\n----- Drive right turn test ------')
        actions = [([0, 0.5, 0], 20), ([0.3, 0.5, 0], 20), ([0.2, 0.5, 0], 10), ([0, 0.1, 0], 30)]
        self._execute_trial(actions, start_track=70)

    def _drive_pass_right_turn_test(self):
        print(f'\n----- Drive pass right turn test ------')
        actions = [([0, 0.3, 0], 80)]
        self._execute_trial(actions, start_track=70)

    def _drive_u_turn_test(self):
        print(f'\n----- Drive u turn test ------')
        actions = [([0, 0.5, 0], 20), ([-0.3, 0.5, 0], 20), ([-0.2, 0.5, 0], 10), ([-0.3, 0.1, 0], 20),
                   ([0, 0.1, 0], 10)]
        self._execute_trial(actions, start_track=103)

    def _drive_pass_u_turn_test(self):
        print(f'\n----- Drive pass u turn test ------')
        actions = [([0, 0.3, 0], 80)]
        self._execute_trial(actions, start_track=103)

    def _drive_s_turn_test(self):
        print(f'\n----- Drive s turn test ------')
        actions = [([0, 0.3, 0], 30), ([-0.4, 0.3, 0], 10), ([0, 0.3, 0], 30), ([0.4, 0.3, 0], 10)]
        self._execute_trial(actions, start_track=250)

    def _drive_pass_s_turn_test(self):
        print(f'\n----- Drive pass s turn test ------')
        actions = [([0, 0.3, 0], 80)]
        self._execute_trial(actions, start_track=250)

    def _execute_trial(self, actions, start_track=0):
        self.environment.is_random_inital_car_position = False
        self._print_total_steps(actions)
        print(f"Action sequence [(action, repetition),...]: {actions}")
        for _ in range(self.trials):
            init_state = self.environment.reset(seed=self.seed)
            self.simulated_environment.reset()
            self._set_car_pos(start_track)
            total_reward_real, total_partial_reward_sim, avg_recon_diff, total_full_sim_reward = self._step(actions, init_state)
            self._print_results(total_reward_real, total_full_sim_reward, total_partial_reward_sim, avg_recon_diff)

    def _step(self, actions, current_state):
        hidden_state = self.simulated_environment.get_hidden_zeros_state()
        total_reward_real = 0
        total_reward_sim = 0
        total_recon_diff = 0
        total_full_sim_reward = 0
        steps = 0
        for (action, repetition) in actions:
            total_full_sim_reward += self._step_sequence_in_dream(actions, current_state, hidden_state)
            for _ in range(repetition):
                current_state, real_reward, _, _ = self.environment.step(action)
                latent_state, vae_reconstruction = self._encode_state(current_state)
                latent_state, simulated_reward, simulated_is_done, hidden_state = self.simulated_environment.step(
                    action, hidden_state, latent_state, is_simulation_real_environment=True)
                mdrnn_reconstruction = self.simulated_environment.current_reconstruction

                self._render(vae_reconstruction)

                total_reward_real += real_reward
                total_reward_sim += simulated_reward
                total_recon_diff += self._compare_reconstructions(vae_reconstruction[0].reshape((64, 64, 3)).numpy(), mdrnn_reconstruction)
                steps += 1

        return total_reward_real, total_reward_sim, total_recon_diff / steps, total_full_sim_reward

    def _step_sequence_in_dream(self, actions, current_state, hidden):
        latent, vae_reconstruction = self._encode_state(current_state)
        total_reward = 0
        for (action, repetition) in actions:
            for _ in range(repetition):
                latent, simulated_reward, simulated_is_done, hidden = self.simulated_environment.step(action, hidden,
                                                                                                      latent,
                                                                                                      is_simulation_real_environment=True)
                total_reward += simulated_reward
                self.simulated_environment.render()
        return total_reward

    def _set_car_pos(self, start_track):
        self.environment.environment.env.car = Car(self.environment.environment.env.world,
                                                   *self.environment.environment.env.track[start_track][1:4])

    def _print_results(self, total_reward_real, total_full_sim_reward, total_partial_reward_sim, avg_recon_diff):
        reward_pct_diff = round((total_reward_real - total_partial_reward_sim) / abs(total_partial_reward_sim) * 100)
        print(
            f'Total real rewards: {total_reward_real}  | Total full sim rewards: {total_full_sim_reward} | Total partial sim rewards: {total_partial_reward_sim} |  Reward Difference: {reward_pct_diff} % '
            f'| MDRNN/VAE avg reconstruction L1 Dist: {round(avg_recon_diff, 4)}')

    def _print_total_steps(self, actions):
        steps = 0
        for (actions, repetition) in actions:
            steps += repetition
        print(f'Total executed steps: {steps}')

    def _compare_reconstructions(self, vae_recon, mdrnn_recon):
        vae_grey = np.mean(vae_recon, axis=2)
        mdrnn_grey = np.mean(mdrnn_recon, axis=2)
        dists = [dist.euclidean(vae_grey[i], mdrnn_grey[i]) for i in range(len(vae_grey))]
        return np.mean(dists)

    def _render(self, vae_reconstruction):
        self.vae_render.render(vae_reconstruction)
        self.environment.render()