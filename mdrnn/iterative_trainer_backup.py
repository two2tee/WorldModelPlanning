""" Iterative trainer trains MDRNN iteratively based on standard or corner rollouts
    Flow:
        Generate k rollouts with planning algorithm i.e., RHEA/RMHC
        Retrain MDRNN with k rollouts for n epochs
        Test MDRNN and store statistics of agent planning performance
        Repeat until l iterations satisfied
"""
#  Copyright (c) 2020, - All Rights Reserved
#  This file is part of the Evolutionary Planning on a Learned World Model thesis.
#  Unauthorized copying of this file, via any medium is strictly prohibited without the consensus of the authors.
#  Written by Thor V.A.N. Olesen <thorolesen@gmail.com> & Dennis T.T. Nguyen <dennisnguyen3000@yahoo.dk>.

import os
import gym
import time
import pickle
import random
import numpy as np
import multiprocessing
from tqdm import tqdm
from PIL import Image
from vae.vae import VAE
from mdrnn.mdrnn import MDRNN
from os.path import join, exists
from vae.vae_trainer import VaeTrainer
from tests.test_suite import PlanningTester
from gym.envs.box2d.car_dynamics import Car
from utility.preprocessor import Preprocessor
from multiprocessing import Pool, Process, Manager
from environment.real_environment import EnvironmentWrapper
from environment.simulated_environment import SimulatedEnvironment


class IterativeTrainer:
    def __init__(self, config, agent, mdrnn_trainer):
        self.config = config
        self.agent = agent
        self.mdrnn_trainer = mdrnn_trainer
        self.threads = multiprocessing.cpu_count()
        self.img_width = config["preprocessor"]["img_width"]
        self.img_height = config["preprocessor"]["img_height"]
        self.max_epochs = config["iterative_trainer"]["max_epochs"]
        self.corner_type = config["iterative_trainer"]['corner_type']  # all, left, right
        self.num_rollouts = config["iterative_trainer"]["num_rollouts"]
        self.data_dir = config["iterative_trainer"]['iterative_data_dir']
        self.num_iterations = config["iterative_trainer"]["num_iterations"]
        self.sequence_length = config["iterative_trainer"]["sequence_length"]
        self.is_corners_only = config["iterative_trainer"]['is_corners_only']
        self.iteration_stats_dir = join(self.config['mdrnn_dir'], 'iteration_stats')

        if not exists(self.iteration_stats_dir):
            os.mkdir(self.iteration_stats_dir)

        if not exists(self.data_dir):
            os.mkdir(self.data_dir)

        print(f'data_dir: {self.data_dir} | cores: {self.threads}')

    def train(self):  # Generate and store planning rollouts -> retrain iterative MDRNN
        print('Start Iterative Training')
        with Manager() as manager:
            iterations_count, test_results, train_losses = self._load_iteration_stats()
            test_results = manager.list(test_results)
            train_losses = manager.list(train_losses)
            test_threads = []
            start_time = time.time()
            print(f'Iterations for model: {iterations_count}')
            progress_bar = tqdm(total=self.num_iterations, desc=f"Current iteration {iterations_count}")
            for i in range(self.num_iterations):
                progress_bar.update(i)
                self._generate_rollouts()  # Generate n planning rollouts of length T
                self._train_mdrnn(train_losses)  # Retrain MRNN with new rollouts
                self._test_planning(iterations_count, test_threads, test_results, train_losses)  # Test plan performance

                if len(test_threads) > 3:  # Wait and Clean up tests processes if too many spawned.
                    for p in test_threads:
                        p.join()
                    test_threads.clear()

                print(f'Iterations for model: {iterations_count} - elapsed {round((time.time() - start_time), 2)}')
                iterations_count += 1

            # ensure all tests are done before exit
            for p in test_threads:
                p.join()

            print('--- Iterative Training Completed ---\n')
            test_results = list(test_results)
            test_results.sort(key=lambda test_result: test_result[0])  # Sort by iterations
            self._save_iteration_stats(iterations_count, test_results, list(train_losses))
            print(test_results)

    def _generate_rollouts(self):
        self.threads = self.num_rollouts if self.num_rollouts < self.threads else self.threads
        rollouts_per_thread = int(self.num_rollouts / self.threads)
        print(f'{self.num_rollouts} rollouts across {self.threads} cores - {rollouts_per_thread} rollouts per thread.')
        with Pool(self.threads) as pool:
            threads = [pool.apply_async(self._rollout_batch, args=(rollouts_per_thread, thread))
                       for thread in range(1, self.threads + 1)]
            [thread.get() for thread in threads]
            pool.close()

        print(f'Done - {self.num_rollouts} rollouts saved in {self.data_dir}')

    def _rollout_batch(self, rollouts_per_thread, thread):
        vae, mdrnn = self._get_vae_mdrnn()
        environment = gym.make("CarRacing-v0")
        agent_wrapper = AgentWrapper(self.agent, self.config, vae, mdrnn)

        for rollout in range(rollouts_per_thread):
            actions, states, rewards, dones = self._corners_rollout(agent_wrapper, environment) if self.is_corners_only else self._standard_rollout(agent_wrapper, environment)
            self._save_rollout(thread, rollout, states, rewards, actions, dones)

        environment.close()

    def _test_planning(self, iteration, test_threads, test_results, train_losses):
        p = Process(target=self._test_thread, args=[iteration, test_results, train_losses])
        p.start()
        test_threads.append(p)

    # ## Needed since multithreading does not work with shared GPU/CPU for model #
    def _train_mdrnn(self, train_losses):
        p = Process(target=self._train_thread, args=[train_losses])
        p.start()
        p.join()

    def _train_thread(self, train_losses):
        vae, mdrnn = self._get_vae_mdrnn()
        self.mdrnn_trainer.sequence_length = self.sequence_length
        _, loss = self.mdrnn_trainer.train(vae, mdrnn, data_dir=self.data_dir, max_epochs=self.max_epochs)
        train_losses.append(loss)

    def _test_thread(self, iteration, test_results, train_losses):
        print(f'Running test for iteration: {iteration}')
        preprocessor = Preprocessor(self.config['preprocessor'])
        vae, mdrnn = self._get_vae_mdrnn()
        environment = EnvironmentWrapper(self.config)  # Set environment
        tester = PlanningTester(self.config, vae, mdrnn, preprocessor, environment, self.agent)
        _, _, trial_rewards, _, _ = tester.run_specific_test("right_turn_planning_test")
        test_reward = (iteration, np.amax(trial_rewards))
        test_results.append(test_reward)
        print(f'Test results for iteration {iteration}: {test_reward}')
        self._save_iteration_stats(iteration, list(test_results), list(train_losses))
        return test_reward

    def _get_vae_mdrnn(self):
        vae_trainer = VaeTrainer(self.config, None, None)
        vae = vae_trainer.reload_model(VAE(self.config))
        mdrnn = MDRNN(num_actions=3,
                      latent_size=self.config['latent_size'],
                      num_gaussians=self.config['mdrnn']['num_gaussians'],
                      num_hidden_units=self.config['mdrnn']['hidden_units'])
        mdrnn = self.mdrnn_trainer.reload_model(mdrnn)
        return vae, mdrnn

    def _standard_rollout(self, agent_wrapper, environment):
        state, environment, _ = self._reset(environment, agent_wrapper)
        actions, states, rewards, dones = [], [], [], []

        for t in range(self.sequence_length):
            action = agent_wrapper.search(state)
            state, reward, done, info = environment.step(action)
            extra_reward = 0.0  # penalize for turning too frequently
            extra_reward -= np.abs(action[0]) / 10.0
            reward += extra_reward
            environment.render()
            environment.env.viewer.window.dispatch_events()
            agent_wrapper.synchronize(state, action)
            state = np.array(Image.fromarray(state).resize(size=(self.img_width, self.img_height)))
            actions.append(action)
            states.append(state)
            rewards.append(reward)
            dones.append(done)
            if done:
                break

        return actions, states, rewards, dones

    def _corners_rollout(self, agent_wrapper, environment):
        actions_rollout, states_rollout, reward_rollout, is_done_rollout = [], [], [], []
        steps = 0
        steps_since_exit = 0
        corners = 0
        is_corner = False
        is_out_corner_pass_checkpoint = True

        #  Search for corner
        state, model, car_position = None, None, None
        while is_corner is False:
            state, model, car_position = self._reset(environment, agent_wrapper)
            for i in range(car_position, len(environment.track)):
                environment.env.car = Car(environment.env.world, *environment.env.track[i][1:4])
                state, reward, done, _ = environment.step([0, 0, 0])
                state = np.array(Image.fromarray(state).resize(size=(self.img_width, self.img_height)))
                if self._is_corner(state, is_in=True):
                    for _ in range(10):
                        state, reward, done, _ = environment.step([0, 0, 0])  # Stabilize car
                    is_corner = True
                    break
        corners += 1
        is_in_corner_pass_checkpoint = False
        #  Record corner
        for t in range(self.sequence_length):
            action = agent_wrapper.search(state)
            state, reward, done, info = environment.step(action)
            agent_wrapper.synchronize(state, action)
            state = np.array(Image.fromarray(state).resize(size=(self.img_width, self.img_height)))

            # Check for corner entry
            if is_in_corner_pass_checkpoint and self._is_corner(state, is_in=True):
                is_corner = True
                is_in_corner_pass_checkpoint = False
                corners += 1
                steps_since_exit = 0
            else:
                is_in_corner_pass_checkpoint = self._corner_in_checkpoint(state)

            if is_corner:
                # Capture data on corner
                actions_rollout.append(action)
                states_rollout.append(state)
                reward_rollout.append(reward)
                is_done_rollout.append(done)

                # Check for corner exit
                if corners > 0:
                    if self._is_corner(state, is_in=False):
                        is_out_corner_pass_checkpoint = False
                    elif not is_out_corner_pass_checkpoint:
                        is_out_corner_pass_checkpoint = self._corner_out_checkpoint(state)
                        if is_out_corner_pass_checkpoint:
                            corners -= 1

                if corners == 0 and steps >= 128 and steps_since_exit > 20 or done:
                    break
                steps_since_exit = steps_since_exit + 1 if corners == 0 else steps_since_exit
                steps += 1
            if not is_corner and t >= int(self.sequence_length/2):  # Early stop if corner not reached
                break
        return actions_rollout, states_rollout, reward_rollout, is_done_rollout

    def _corner_in_checkpoint(self, frame):
        return not self._is_corner(frame, is_in=True)

    def _corner_out_checkpoint(self, frame):
        return not self._is_corner(frame, is_in=False)

    def _is_corner(self, frame, is_in):
        y = 0 if is_in else 55  # top else bot
        pixel_row = frame[y][:][:]

        for i, pixel in enumerate(pixel_row):
            if self._is_white_pixel(pixel) or self._is_red_pixel(pixel):
                previous_pixel = pixel_row[i-2]
                if self.corner_type == 'right' and self._is_grey_pixel(previous_pixel):
                    return True
                if self.corner_type == 'left' and not self._is_grey_pixel(previous_pixel):
                    return True
                if self.corner_type == 'all':
                    return True
        return False

    def _is_grey_pixel(self, pixel):
        return pixel[0] < 115 and pixel[1] < 115 and pixel[2] < 115

    def _is_white_pixel(self, pixel):
        return pixel[0] > 140 and pixel[1] > 140 and pixel[2] > 140

    def _is_red_pixel(self, pixel):
        return pixel[0] > 150 and pixel[1] < 200 and pixel[2] < 200

    def _save_rollout(self, thread, rollout_number, states, rewards, actions, dones):
        print(f"Thread {thread} - End of rollout {rollout_number}, {len(states)} frames.")
        print(self.data_dir, f'thread_{thread}_rollout_{rollout_number}')
        np.savez_compressed(file=join(self.data_dir, f'iterative_thread_{thread}_rollout_{rollout_number}'),
                            observations=np.array(states),
                            rewards=np.array(rewards),
                            actions=np.array(actions),
                            terminals=np.array(dones))

    def _reset(self, environment, agent_wrapper):
        seed = random.randint(0, 2 ** 31 - 1)
        environment.seed(seed)
        agent_wrapper.reset()
        environment.reset()
        obs = [environment.env.step([0, 0, 0])[0] for _ in range(50)][-1]
        car_position = np.random.randint(len(environment.env.track))
        environment.car = Car(environment.world, *environment.track[car_position][1:4])
        environment.viewer.window.dispatch_events()  # Garbage collection of events in viewer
        return obs, environment, car_position

    def _save_iteration_stats(self, iterations, test_results, train_loss):
        stats_filename = f'iterative_stats_{self.config["experiment_name"]}'
        stats_filepath = join(self.iteration_stats_dir, f'{stats_filename}')

        stats_data = {'iterations': iterations, 'test_results': test_results, 'train_loss': train_loss}

        with open(f'{stats_filepath}.pickle', 'wb') as file:
            pickle.dump(stats_data, file, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_iteration_stats(self):
        stats_filename = f'iterative_stats_{self.config["experiment_name"]}'
        stats_filepath = join(self.iteration_stats_dir, f'{stats_filename}.pickle')
        mdrnn_filename = f"{self.config['mdrnn_dir']}/checkpoints/iterative_{self.config['experiment_name']}_{self.config['mdrnn_trainer']['mdrnn_best_filename']}"

        if exists(stats_filepath) and exists(mdrnn_filename):
            with open(f'{stats_filepath}', 'rb') as file:
                stats_data = pickle.load(file)
                return stats_data['iterations'], stats_data['test_results'], stats_data['train_loss']
        return 0, [], []


class AgentWrapper:
    def __init__(self, agent, config, vae, mdrnn):
        self.agent = agent
        self.vae = vae
        self.config = config
        self.mdrnn = mdrnn
        self.preprocessor = Preprocessor(config['preprocessor'])
        self.simulated_environment = SimulatedEnvironment(config, vae, mdrnn)

        self.hidden = self.simulated_environment.get_hidden_zeros_state()
        self.latent = None

    def search(self, state):
        _, self.latent = self._reconstruct(state)
        if self.config['planning']['planning_agent'] == "MCTS":
            action = self.agent.search(self.simulated_environment, self.latent, self.hidden)
        else:
            action, _ = self.agent.search(self.simulated_environment, self.latent, self.hidden)
        return action

    def synchronize(self, next_state, action):
        reconstruction, latent_state = self._reconstruct(next_state)
        self.latent, _, _, self.hidden = self.simulated_environment.step(action, self.hidden, latent_state,
                                                                         is_simulation_real_environment=True)

    def reset(self):
        self.latent = None
        self.hidden = self.simulated_environment.get_hidden_zeros_state()
        self.simulated_environment.reset()

    def _reconstruct(self, state):
        state = self.preprocessor.resize_frame(state).unsqueeze(0)
        reconstruction, z_mean, z_log_standard_deviation = self.vae(state)
        latent_state = self.vae.sample_reparametarization(z_mean, z_log_standard_deviation)
        return reconstruction, latent_state




