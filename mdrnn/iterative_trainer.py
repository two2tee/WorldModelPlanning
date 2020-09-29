""" Iterative trainer trains MDRNN iteratively based on planning rollouts
    1) Generate n=10,000 rollouts with RHEA planning algorithm
    2) Retrain MDRNN on n planning rollouts for k=4 epochs
    3) Test MDRNN and store statistics of agent planning performance
    4) Repeat until l iterations satisfied or reward 900 across 100 games completed
"""
#  Copyright (c) 2020, - All Rights Reserved
#  This file is part of the Evolutionary Planning on a Learned World Model thesis.
#  Unauthorized copying of this file, via any medium is strictly prohibited without the consensus of the authors.
#  Written by Thor V.A.N. Olesen <thorolesen@gmail.com> & Dennis T.T. Nguyen <dennisnguyen3000@yahoo.dk>.

import os
import sys

import gym
import time
import pickle
import platform
import numpy as np
import multiprocessing
from copy import copy

import torch
from gym.envs.box2d.car_dynamics import Car
from tqdm import tqdm
from PIL import Image
from vae.vae import VAE
from mdrnn.mdrnn import MDRNN
from os.path import join, exists
from vae.vae_trainer import VaeTrainer
from utility.preprocessor import Preprocessor
from tests_custom.test_suite_factory import get_planning_tester
from planning.simulation.agent_wrapper import AgentWrapper
from environment.environment_factory import get_environment
from utility.logging.planning_logger import PlanningLogger
from torch.multiprocessing import Pool, Process, Manager, RLock, Lock, Value
from mdrnn.iteration_stats.iteration_result import IterationResult
from environment.actions.action_sampler_factory import get_action_sampler
gym.logger.set_level(40)  # Disable user warnings

if platform.system() == "Darwin" or platform.system() == "Linux":
    print("Spawn method enabled over fork on Mac OSX / Linux")
    multiprocessing.set_start_method("spawn", force=True)


class IterativeTrainer:
    def __init__(self, config, planning_agent, mdrnn_trainer):
        self.config = config
        self.mdrnn_trainer = mdrnn_trainer
        self.planning_agent = planning_agent
        self.threads = self._get_threads()
        self.img_width = config["preprocessor"]["img_width"]
        self.img_height = config["preprocessor"]["img_height"]
        self.max_epochs = config["iterative_trainer"]["max_epochs"]
        self.num_rollouts = config["iterative_trainer"]["num_rollouts"]
        self.test_scenario = config["iterative_trainer"]["test_scenario"]
        self.data_dir = config["iterative_trainer"]['iterative_data_dir']
        self.num_iterations = config["iterative_trainer"]["num_iterations"]
        self.sequence_length = config["iterative_trainer"]["sequence_length"]
        self.iteration_stats_dir = join(self.config['mdrnn_dir'], 'iteration_stats')
        self.max_test_threads = config["iterative_trainer"]["max_test_threads"]
        self.max_test_threads = self.threads if self.max_test_threads > self.threads else self.max_test_threads
        self.is_replay_buffer = config["iterative_trainer"]["replay_buffer"]['is_replay_buffer']
        self.max_buffer_size = config["iterative_trainer"]["replay_buffer"]['max_buffer_size']
        self._rollout_counter = self._get_rollout_file_count()
        if not exists(self.iteration_stats_dir):
            os.mkdir(self.iteration_stats_dir)

        if not exists(self.data_dir):
            os.mkdir(self.data_dir)

        print(f'data_dir: {self.data_dir} | cores used: {self.threads}')

    def train(self):
        print('Started Iterative Training')
        with Manager() as manager:
            iteration_results, iterations_count = self._load_iteration_stats()
            iteration_results = manager.dict(iteration_results)
            test_threads = []
            start_time = time.time()

            # if iterations_count == 0:  # Pre-test non iterative model for baseline
            #     iteration_results[iterations_count] = IterationResult(iteration=0)
            #     self._test_planning(iteration=0, iteration_results=iteration_results, test_threads=test_threads)

            for _ in tqdm(range(self.num_iterations), desc=f"Current iteration {iterations_count+1}"):  # TODO: replace with "while task (900+ reward) is not completed"
                iterations_count += 1
                iteration_results[iterations_count] = IterationResult(iteration=iterations_count)

                self._generate_rollouts(iterations_count)  # Generate n planning rollouts of length T
                self._train_mdrnn(copy(iterations_count), iteration_results)
                # self._test_planning(iterations_count, iteration_results, test_threads)  # Test plan performance

                print(f'Iterations for model: {iterations_count} - {round((time.time() - start_time), 2)} seconds')

            [p.join() for p in test_threads]  # ensure all tests are done before exit
            print('--- Iterative Training Completed ---\n')
            self._save_iteration_stats(iteration_results)

    def _generate_rollouts(self, iteration):
        vae, mdrnn = self._get_vae_mdrnn()
        vae, mdrnn = vae.eval(), mdrnn.eval()
        self.threads = self.num_rollouts if self.num_rollouts < self.threads else self.threads
        self._set_torch_threads(threads=1)  # 1 to ensure underlying threads only uses 1 thread to prevent hidden threading
        num_rollouts_per_thread = int(self.num_rollouts / self.threads)
        print(f'{self.num_rollouts} rollouts across {self.threads} cores with {num_rollouts_per_thread} rollouts each.')

        shared_rollout_counter = Value('i', self._rollout_counter,)
        with Pool(int(self.threads), initargs=(Lock(), RLock(), shared_rollout_counter,), initializer=self.init_globals) as pool:
            threads = [pool.apply_async(self._get_rollout_batch, args=(num_rollouts_per_thread, thread_id, iteration, vae, mdrnn))
                       for thread_id in range(1, self.threads + 1)]
            pool.close()
            [thread.get() for thread in threads]
            pool.close()
        self._rollout_counter = shared_rollout_counter.value

        print(f'Done - {self.num_rollouts} rollouts saved in {self.data_dir}')

    def init_globals(self, _rollout_lock, tqdm_lock, _rollout_counter):
        global rollout_lock
        global rollout_counter
        rollout_lock = _rollout_lock
        rollout_counter = _rollout_counter
        tqdm.set_lock(tqdm_lock)

    def _set_rollout_count(self):
        if self.is_replay_buffer:
            rollout_counter.value = 1 if rollout_counter.value > self.max_buffer_size else rollout_counter.value + 1
        else:
            rollout_counter.value = 1 if rollout_counter.value > self.num_rollouts else rollout_counter.value + 1

    def _get_rollout_file_count(self):
        return len([name for root, dirs, files in os.walk(self.data_dir) for name in files])

    def _get_rollout_batch(self, num_rollouts_per_thread, thread_id, iteration, vae, mdrnn):
        environment = get_environment(self.config)
        agent_wrapper = AgentWrapper(self.planning_agent, self.config, vae, mdrnn)
        for rollout_number in range(1, num_rollouts_per_thread + 1):
            actions, states, rewards, terminals = self._create_rollout(agent_wrapper, environment, thread_id, rollout_number, num_rollouts_per_thread, iteration)

            with rollout_lock:
                self._set_rollout_count()
                self._save_rollout(actions, states, rewards, terminals)

        environment.close()

    def _test_planning(self, iteration, iteration_results, test_threads):
        if len(test_threads) >= self.max_test_threads:  # Prevent spawning too many test threads
            [p.join() for p in test_threads]
        p = Process(target=self._test_thread, args=[iteration, iteration_results])
        p.start()
        test_threads.append(p)

    # Needed since multi threading does not work with shared GPU/CPU for model
    def _train_mdrnn(self, iteration, iteration_results):
        start = time.time()
        self._set_torch_threads(threads=multiprocessing.cpu_count())
        p = Process(target=self._train_thread, args=[iteration, iteration_results])
        p.start()
        p.join()
        print(f"TOTAL TRAINING TIME: {round(time.time() - start, 1)} seconds")

    def _train_thread(self, iteration, iteration_results):
        vae, mdrnn = self._get_vae_mdrnn()
        _, test_losses = self.mdrnn_trainer.train(vae, mdrnn, data_dir=self.data_dir, max_epochs=self.max_epochs,
                                                  seq_len=self.sequence_length, iteration=iteration,
                                                  max_size=self.num_rollouts, random_sampling=self.is_replay_buffer)
        iteration_result = iteration_results[iteration]
        iteration_result.mdrnn_test_losses = test_losses
        iteration_results[iteration] = iteration_result

    def _test_thread(self, iteration, iteration_results):
        print(f'Running test for iteration: {iteration}')
        preprocessor = Preprocessor(self.config['preprocessor'])
        vae, mdrnn = self._get_vae_mdrnn()
        tester = get_planning_tester(self.config, vae, mdrnn, preprocessor, self.planning_agent)

        session_name = self._make_session_name(self.config["experiment_name"], self.config['planning']['planning_agent'], iteration)
        test_name, trials_actions, trials_rewards, trials_elites, trial_max_rewards, trial_seeds = tester.run_specific_test(self.test_scenario, session_name)

        iteration_result = iteration_results[iteration]
        iteration_result.agent_name = self.config['planning']['planning_agent']
        iteration_result.test_name = test_name
        iteration_result.trial_seeds = trial_seeds
        iteration_result.total_trials = len(trial_max_rewards)
        iteration_result.trials_rewards = trials_rewards
        iteration_result.trials_max_rewards = trial_max_rewards
        iteration_results[iteration] = iteration_result
        self._log_iteration_test_results(iteration_result)
        self._save_iteration_stats(iteration_results)
        print(f'Test for iteration: {iteration} completed')
        return test_name

    def _get_vae_mdrnn(self):
        vae_trainer = VaeTrainer(self.config, preprocesser=None)
        vae = vae_trainer.reload_model(VAE(self.config), device='cpu')
        vae.cpu()
        mdrnn = MDRNN(num_actions=get_action_sampler(self.config).num_actions,
                      latent_size=self.config['latent_size'],
                      num_gaussians=self.config['mdrnn']['num_gaussians'],
                      num_hidden_units=self.config['mdrnn']['hidden_units'])
        mdrnn = self.mdrnn_trainer.reload_model(mdrnn, device='cpu')
        mdrnn.cpu()
        return vae, mdrnn

    def _create_rollout(self, agent_wrapper, environment, thread_id, rollout_number, num_rollouts_per_thread, iteration):
        state, environment = self._reset(environment, agent_wrapper)
        actions, states, rewards, terminals = [], [], [], []
        progress_description = f"Data generation at iteration {iteration} | thread: {thread_id} | rollout: {rollout_number}/{num_rollouts_per_thread}"
        for _ in tqdm(range(self.sequence_length+1), desc=progress_description, position=thread_id-1):
            action = agent_wrapper.search(state)
            state, reward, done, info = environment.step(action)
            agent_wrapper.synchronize(state, action)
            state = np.array(Image.fromarray(state).resize(size=(self.img_width, self.img_height)))
            actions.append(action)
            states.append(state)
            rewards.append(reward)
            terminals.append(done)
        return actions, states, rewards, terminals

    def _set_car_position(self, start_track, environment):
        if start_track == 1:
            return
        environment.environment.env.car = Car(environment.environment.env.world, *environment.environment.env.track[start_track][1:4])

    def _save_rollout(self, actions, states, rewards, terminals):
        file_name = f'iterative_rollout_{rollout_counter.value}'
        np.savez_compressed(file=join(self.data_dir, file_name),
                            observations=np.array(states),
                            rewards=np.array(rewards),
                            actions=np.array(actions),
                            terminals=np.array(terminals))

    def _reset(self, environment, agent_wrapper):
        agent_wrapper.reset()
        obs = environment.reset(seed=2)
        self._set_car_position(start_track=222, environment=environment)
        return obs, environment

    def _log_iteration_test_results(self, iteration_result):
        logger = PlanningLogger(is_logging=True)
        logger.start_log(name=f'{self.config["experiment_name"]}_main_{iteration_result.agent_name}')

        logger.log_iteration_max_reward(test_name=iteration_result.test_name, trials=iteration_result.total_trials,
                                             iteration=iteration_result.iteration, max_reward=iteration_result.get_average_max_reward())
        logger.log_iteration_avg_reward(test_name=iteration_result.test_name, trials=iteration_result.total_trials,
                                        iteration=iteration_result.iteration, avg_reward=iteration_result.get_average_total_reward())
        logger.log_reward_mean_std(iteration_result.test_name, iteration_result.trials_rewards, iteration_result.iteration)
        logger.end_log()


    def _save_iteration_stats(self, iteration_results):
        stats_filename = f'iterative_stats_{self.config["experiment_name"]}'
        stats_filepath = join(self.iteration_stats_dir, f'{stats_filename}')

        encoded_iteration_results = [iteration_result.to_dict() for iteration_result in list(iteration_results.values())
                                     if iteration_result.test_name]  # Only save completed iteration

        file_content = {'iteration_results': encoded_iteration_results}
        with open(f'{stats_filepath}.pickle', 'wb') as file:
            pickle.dump(file_content, file, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_iteration_stats(self):
        stats_filename = f'iterative_stats_{self.config["experiment_name"]}'
        stats_filepath = join(self.iteration_stats_dir, f'{stats_filename}.pickle')
        mdrnn_filename = f"{self.config['mdrnn_dir']}/checkpoints/iterative_{self.config['experiment_name']}_{self.config['mdrnn_trainer']['mdrnn_best_filename']}"

        if exists(stats_filepath) and exists(mdrnn_filename):
            with open(f'{stats_filepath}', 'rb') as file:
                file_content = pickle.load(file)
                decoded_results = [IterationResult.to_obj(encoded_result) for encoded_result in file_content['iteration_results']]
                iteration_results = {}
                for iteration_result in decoded_results:
                    iteration_results[iteration_result.iteration] = iteration_result
                return iteration_results, len(iteration_results)
        return {}, 0

    def _get_threads(self):
        fixed_cores = self.config["iterative_trainer"]["fixed_cpu_cores"]
        return fixed_cores if fixed_cores else multiprocessing.cpu_count()

    def _set_torch_threads(self, threads):
        torch.set_num_threads(threads)
        os.environ['OMP_NUM_THREADS'] = str(threads)  # Inference in CPU to avoid cpu scheduling - slow parallel data generation

    def _make_session_name(self, model_name, agent_name,  iteration):
        return f'{model_name}_{agent_name}_iteration_{iteration}_h{self.planning_agent.horizon}_g{self.planning_agent.max_generations}_sb{self.planning_agent.is_shift_buffer}'

