""" Test suite with support for planning test/benchmarking and World Model tests (i.e., reward signals and reconstructions) """
#  Copyright (c) 2020, - All Rights Reserved
#  This file is part of the Evolutionary Planning on a Learned World Model thesis.
#  Unauthorized copying of this file, via any medium is strictly prohibited without the consensus of the authors.
#  Written by Thor V.A.N. Olesen <thorolesen@gmail.com> & Dennis T.T. Nguyen <dennisnguyen3000@yahoo.dk>.

import os
import time
import torch
import pickle
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
from pprint import pprint
from datetime import date
from os.path import join, exists
from utility.visualizer import Visualizer
from gym.envs.box2d.car_dynamics import Car
from concurrent.futures import ProcessPoolExecutor
from environment.simulated_environment import SimulatedEnvironment
from planning.simulation.mcts_simulation import MCTS as MCTS_simulation
from planning.simulation.rolling_horizon_simulation import RHEA as RHEA_simulation
from planning.simulation.random_mutation_hill_climbing_simulation import RMHC as RMHC_simulation

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


# ARGS KEYS PLANNING
ACTION_HISTORY = 'action_history'
ELITES = 'elites'
OPTIMAL_STEPS = 'optimal_steps'
OPTIMAL_REWARD = 'optimal_reward'
RANDOM_REWARD = 'random_reward'
TILES_TO_COMPLETE = 'tiles_to_complete'
CUSTOM_SEED = 'custom_seed'
TEST_NAME = 'test_name'


class PlanningTester(BaseTester):
    def __init__(self, config, vae, mdrnn, preprocessor, environment, planning_agent):
        super().__init__(config, vae, mdrnn, preprocessor, environment, trials=config["test_suite"]["trials"])
        self.is_multithread = self.config['test_suite']['is_multithread']
        self.planning_agent = planning_agent
        self.planning_dir = join('tests', config['test_suite']['planning_test_log_dir'])
        self.car_forward_velocity_approx = 0
        self.velocity_growth_penalizer = 0.005
        self.visualizer = Visualizer()
        self.is_ntbea_tuning = False
        self.is_render = config['visualization']['is_render']
        self.is_render_best_elite_only = self.config['visualization']['is_render_best_elite_only']
        self.is_render_fitness = self.config['visualization']['is_render_fitness']
        self.is_render_trajectory = self.config['visualization']['is_render_trajectory']
        self.is_render_simulation = self.config['visualization']['is_render_simulation']
        self.is_render_dream = self.config['visualization']['is_render_dream']
        torch.set_num_threads(1)
        os.environ['OMP_NUM_THREADS'] = '1'

    def get_test_functions(self):
        return {  # (test_func, args)
                "forward_planning_test": (self._planning_forward_test, {OPTIMAL_REWARD: 66, OPTIMAL_STEPS: 100, RANDOM_REWARD: -7, TILES_TO_COMPLETE: 25, CUSTOM_SEED: 9214}),
                "left_turn_planning_test": (self._planning_left_turn_test, {OPTIMAL_REWARD: 23, OPTIMAL_STEPS: 60, RANDOM_REWARD: -3, TILES_TO_COMPLETE: 10, CUSTOM_SEED: 9214}),
                "right_turn_planning_test": (self._planning_right_turn_test,{OPTIMAL_REWARD: 33, OPTIMAL_STEPS: 60, RANDOM_REWARD: -3, TILES_TO_COMPLETE: 11, CUSTOM_SEED: 2}),
                "s_turn_planning_test": (self._planning_s_turn_test, {OPTIMAL_REWARD: 43, OPTIMAL_STEPS: 80, RANDOM_REWARD: -3, TILES_TO_COMPLETE: 16, CUSTOM_SEED: 9214}),
                "u_turn_planning_test": (self._planning_u_turn_test,{OPTIMAL_REWARD: 40, OPTIMAL_STEPS: 80, RANDOM_REWARD: -5, TILES_TO_COMPLETE: 15, CUSTOM_SEED: 9214}),
                "planning_whole_track_no_right_turns_test": (self._planning_whole_track_no_right_turns_test, {OPTIMAL_REWARD: 900, OPTIMAL_STEPS: 1200, RANDOM_REWARD: -32, TILES_TO_COMPLETE: 1200, CUSTOM_SEED: 30}),
                "planning_whole_random_track": (self._planning_whole_random_track_test, {OPTIMAL_REWARD: 900, OPTIMAL_STEPS: 1200, RANDOM_REWARD: -3, TILES_TO_COMPLETE: 1200, CUSTOM_SEED: None})
                }

    def run_tests(self):
        print('------- Planning Tests params --------')
        print(f'seed: {self.seed} | trials per test: {self.trials}')
        print(f'Planning Agent: {type(self.planning_agent)}')
        pprint(vars(self.planning_agent))  # Print agent params

        if self.config['test_suite']["is_reload_planning_session"] and self._is_session_exists():
            return self._run_cached_session()
        else:
            return self._run_new_session()

    def run_specific_test(self, test_name):
        with torch.no_grad():
            test_func, args = self.get_test_functions()[test_name]
            trial_actions, trial_rewards, trial_elites, trial_max_rewards, seed = test_func(args=args)
            plt.close('all')
            return test_name, trial_actions, trial_rewards, trial_elites, trial_max_rewards, seed

    def _run_new_session(self):
        print('\n--- RUNNING NEW PLANNING TESTS ---')
        return self._run_multithread_new_test_session() if self.is_multithread and not self.is_render else \
               self._run_singlethread_new_test_session()

    def _run_multithread_new_test_session(self):
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            tests = self.get_test_functions()
            thread_results = list(executor.map(self.run_specific_test, tests.keys()))
            test_results = {}
            for test_name, trial_actions, trial_rewards, trial_elites, trial_max_rewards, seed in thread_results:
                test_results[test_name] = (test_name, trial_actions, trial_rewards, trial_elites, seed)
                if len(trial_rewards) > 1:
                    print(f'{self.config["planning"]["planning_agent"]} - {test_name} - Average reward over {len(trial_rewards)} trials: {np.mean(trial_rewards)}')
            self._save_test_session(test_results)
            return self._get_session_total_best_reward(test_results)

    def _run_singlethread_new_test_session(self):
        test_results = {}
        tests = self.get_test_functions()
        with torch.no_grad():
            for test_name in tests.keys():
                test_func, args = tests[test_name]
                trial_actions, trial_rewards, trial_elites, trial_max_rewards, seed = test_func(args=args)
                test_results[test_name] = (test_name, trial_actions, trial_rewards, trial_elites, seed)
                print(f'Average reward over {len(trial_rewards)} trials: {np.mean(trial_rewards)}')
            self._save_test_session(test_results)
        plt.close('all')
        return self._get_session_total_best_reward(test_results)

    def _run_cached_session(self):
        print('--- RUNNING CACHED PLANNING TESTS ---')
        session = self._load_test_session(session_name=self.config['test_suite']['planning_session_to_load'])
        print(session)
        tests = self.get_test_functions()
        session_reward = 0
        for test_name in session.keys():
            best_trial, best_actions, best_reward, elites, seed = self._get_best_trial_action_and_reward(session[test_name])  # ONLY RUN BEST TRIALS
            print(f'Reload actions from {test_name}, trial {best_trial} with best reward {best_reward}')
            test_func, args = tests[test_name]
            args[ACTION_HISTORY] = best_actions
            args[ELITES] = elites
            args[CUSTOM_SEED] = seed if seed is not None else args[CUSTOM_SEED]
            _, reward = test_func(args=args)
            session_reward += reward
        print(f'Total session reward: {session_reward}')
        return self._get_session_total_best_reward(session)


    # TEST METHODS #######################################

    def _planning_forward_test(self, args):
        args[TEST_NAME] = '----- Forward planning test ------'
        return self._run_plan_or_replay(start_track=25, args=args)

    def _planning_left_turn_test(self, args):
        args[TEST_NAME] = '----- Left Turn planning test ------'
        return self._run_plan_or_replay(start_track=14, args=args)

    def _planning_right_turn_test(self, args):
        args[TEST_NAME] = '----- Right Turn planning test ------'
        return self._run_plan_or_replay(start_track=222, args=args)

    def _planning_u_turn_test(self, args):
        args[TEST_NAME] = '----- U-Turn planning test ------'
        return self._run_plan_or_replay(start_track=103, args=args)

    def _planning_s_turn_test(self, args):
        args[TEST_NAME] = '----- S-Turn planning test ------'
        return self._run_plan_or_replay(start_track=250, args=args)

    def _planning_whole_track_no_right_turns_test(self, args):
        args[TEST_NAME] = '----- Whole track (No right turns) planning test ------'
        return self._run_plan_or_replay(start_track=1, args=args)

    def _planning_whole_random_track_test(self, args):
        args[TEST_NAME] = '----- Whole random track planning test ------'
        return self._run_plan_or_replay(start_track=1, args=args)

    # ######################################################

    def _run_plan_or_replay(self, start_track, args):
        print(args[TEST_NAME])
        if ACTION_HISTORY in args:
            return self._replay_planning_test(start_track, args)
        else:
            return self._run_planning_test(start_track, args)

    def _run_planning_test(self, start_track, args):
        trial_actions = []
        trial_rewards = []
        trial_max_rewards = []
        trial_elites = []

        seed = args[CUSTOM_SEED]

        for i in range(self.trials):
            self.environment.reset(seed)
            seed = self.environment.seed
            if args[CUSTOM_SEED] is not None:
                self._set_car_position(start_track)
            current_state = self._skip_zoom()

            self.simulated_environment.reset()
            latent_state, _ = self._encode_state(current_state)
            hidden_state = self.simulated_environment.get_hidden_zeros_state()

            trial_results_dto = self._get_trial_results_dto(args)
            elites = []
            action_history = []
            start_time = time.time()
            total_reward = 0
            steps_ran = 0
            elapsed_time = 0

            for step in range(args['optimal_steps']+75):
                if self.config['planning']['planning_agent'] == "MCTS":
                    action = self.planning_agent.search(self.simulated_environment, latent_state, hidden_state)
                else:
                    action, step_elites = self.planning_agent.search(self.simulated_environment, latent_state, hidden_state)
                    elites.append(step_elites)
                    if self.is_render:
                        if self.is_render_fitness:
                            self.visualizer.show_fitness_plot(self.planning_agent.max_generations, step_elites, self.config['planning']['planning_agent'])
                        if self.is_render_trajectory:
                            self.visualizer.show_trajectory_plot(current_state, step_elites, self.config['planning']['planning_agent'], self.environment.environment, self.is_render_best_elite_only)

                if self.is_render_dream:
                    self._step_sequence_in_dream(self.planning_agent.current_elite.action_sequence, current_state, hidden_state)

                current_state, reward, is_done, simulated_reward, simulated_is_done, latent_state, hidden_state = self._step(action, hidden_state)
                action_history.append(action)
                total_reward += reward
                steps_ran += 1
                elapsed_time = time.time() - start_time
                self._update_trial_results(trial_results_dto, reward, total_reward, steps_ran)

            trial_elites.append(elites)
            trial_actions.append(action_history)
            trial_rewards.append(total_reward)
            trial_max_rewards.append(trial_results_dto['max_reward'])

            self._print_trial_results(i, elapsed_time, total_reward, steps_ran, trial_results_dto)

        return trial_actions, trial_rewards, trial_elites, trial_max_rewards, seed

    def _replay_planning_test(self, start_track, args):
        actions = args[ACTION_HISTORY]
        elites = args[ELITES]
        seed = args[CUSTOM_SEED]

        _ = self.environment.reset(seed=seed)
        self.simulated_environment.reset()
        hidden_state = self.simulated_environment.get_hidden_zeros_state()
        self._set_car_position(start_track)
        self._skip_zoom()

        trial_results_dto = self._get_trial_results_dto(args)

        steps_ran = 0
        total_reward = 0
        for i, action in enumerate(actions):
            action = self._get_action(action)
            current_state, reward, is_done, simulated_reward, simulated_is_done, latent_state, hidden_state = self._step(action, hidden_state)
            steps_ran += 1
            total_reward += reward
            self._update_trial_results(trial_results_dto, reward, total_reward, steps_ran)

            if elites and self.is_render:
                if self.is_render_trajectory:
                    self.visualizer.show_trajectory_plot(current_state, elites[i], self.config['planning']['planning_agent'], self.environment.environment, self.is_render_best_elite_only)

                if self.is_render_fitness:
                    self.visualizer.show_fitness_plot(len(elites[i])-1, elites[i], self.config['planning']['planning_agent'])

                if self.is_render_dream:
                    self._step_sequence_in_dream(elites[i][-1][2], current_state, hidden_state)

        self._print_trial_results(None, None, total_reward, steps_ran, trial_results_dto)

        return actions, total_reward

    def _step(self, action, hidden_state):
        current_state, reward, is_done, _ = self.environment.step(action)
        latent_state, _ = self._encode_state(current_state)
        latent_state, simulated_reward, simulated_is_done, hidden_state = self.simulated_environment.step(action, hidden_state, latent_state, is_simulation_real_environment=True)
        if self.is_render and self.is_render_simulation:
            self.simulated_environment.render()
        self.environment.render()
        return current_state, reward, is_done, simulated_reward, simulated_is_done, latent_state, hidden_state

    def _step_sequence_in_dream(self, actions, current_state, hidden):
        latent, _ = self._encode_state(current_state)
        total_reward = 0

        for action in actions:
            latent, simulated_reward, simulated_is_done, hidden = self.simulated_environment.step(self._get_action(action), hidden, latent, is_simulation_real_environment=True)
            total_reward += simulated_reward
            self.simulated_environment.render()
        return total_reward

    def _get_action(self, action):
        return action[0] if type(action) == tuple and len(action) == 2 else action

    def _set_car_position(self, start_track):
        self.environment.environment.env.car = Car(self.environment.environment.env.world,
                                                   *self.environment.environment.env.track[start_track][1:4])

    def _skip_zoom(self):
        state = None
        for _ in range(50):
            state, _, _, _ = self.environment.step([0, 0, 0])
        self.environment.render()
        return state

    def _is_session_exists(self):
        filename = join(self.planning_dir, self.config['test_suite']['planning_session_to_load'])+'.pickle'
        is_exists = exists(filename)
        print(f'\nFOUND SESSION: {filename}') if is_exists else print(f'COULD NOT FIND SESSION {filename}')
        return is_exists

    def _get_best_trial_action_and_reward(self, test_result):
        seed = None
        if len(test_result) is 5:
            test_name, trial_actions, trial_rewards, trial_elites, seed = test_result
        else:
            test_name, trial_actions, trial_rewards, trial_elites = test_result

        index_max = np.argmax(trial_rewards)
        return index_max+1, trial_actions[index_max], trial_rewards[index_max], trial_elites[index_max], seed

    def _get_session_total_best_reward(self, test_results):
        total_reward = 0
        for test_name, actions, rewards, elites, seed in test_results.values():
            total_reward += np.amax(rewards)
        return total_reward

    def _get_agent_parameters(self, agent_type):
        agent_params = {
            'RHEA': self.config['planning']['rolling_horizon'],
            'RMHC': self.config['planning']['random_mutation_hill_climb'],
            "MCTS": self.config['planning']['monte_carlo_tree_search'],
        }
        params = {'agent_parameters': agent_params[agent_type]}
        if agent_type == 'RHEA' or agent_type == 'RMHC':
            params['evolution_settings'] = self.config['evolution_handler']

        return params

    def _restore_agent(self, agent_type, params):
        if agent_type == "RHEA":
            return RHEA_simulation(*params.values())
        elif agent_type == "RMHC":
            return RMHC_simulation(*params.values())
        elif agent_type == "MCTS":
            return MCTS_simulation(*params.values())
        else:
            raise Exception(f'Invalid agent type: {agent_type}')

    def _save_test_session(self, test_results):
        save_data = {
            'agent_type': self.config['planning']['planning_agent'],
            'agent_params': self._get_agent_parameters(self.config['planning']['planning_agent']),
            'test_results': test_results
        }
        if not exists(self.planning_dir):
            os.mkdir(self.planning_dir)
        session_reward = round(self._get_session_total_best_reward(test_results), 4)
        session_date = date.today().strftime('%Y-%m-%d-%H.%M')

        file_name = f'{"NTBEA_" if self.is_ntbea_tuning else ""}{save_data["agent_type"]}_{self.config["experiment_name"]}_planning_session_{session_date}_total_session_best_reward_{session_reward}'
        path = join(self.planning_dir, file_name)
        with open(f'{path}.pickle', 'wb') as file:
            pickle.dump(save_data, file, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'results saved at: {path}.pickle')

    def _load_test_session(self, session_name):
        file_name = join(self.planning_dir, session_name)
        with open(f'{file_name}.pickle', 'rb') as file:
            save_data = pickle.load(file)
            print(save_data['agent_params'])
            test_results = save_data['test_results']
            return test_results

    def _reward_diff_percentage(self, actual, control):
        return round((actual-control) / 100) if abs(actual) == 0 else round((actual-control) / abs(actual) * 100)

    def _get_trial_results_dto(self, args):
        return {
            'test_name': args[TEST_NAME],
            'reward_at_optimum_steps': 0,
            'reward_at_success': 0,
            'steps_at_success': 0,
            'tiles': 0,
            'test_success': False,
            'max_reward': 0,
            'optimal_steps': args[OPTIMAL_STEPS],
            'optimal_reward': args[OPTIMAL_REWARD],
            'random_reward': args[RANDOM_REWARD],
            'tiles_to_complete': args[TILES_TO_COMPLETE]
        }

    def _update_trial_results(self, trial_results_dto, reward, total_reward, steps_ran):
        tiles = trial_results_dto['tiles']
        tiles_to_complete = trial_results_dto['tiles_to_complete']
        reward_at_success = trial_results_dto['reward_at_success']
        reward_at_optimum_steps = trial_results_dto['reward_at_optimum_steps']
        optimal_steps = trial_results_dto['optimal_steps']

        trial_results_dto['max_reward'] = total_reward if total_reward > trial_results_dto['max_reward'] else trial_results_dto['max_reward']

        trial_results_dto['tiles'] = tiles + 1 if reward > 0 and tiles <= tiles_to_complete else tiles
        if tiles <= tiles_to_complete:
            trial_results_dto['test_success'] = tiles == tiles_to_complete
            trial_results_dto['steps_at_success'] = steps_ran if tiles_to_complete == tiles_to_complete else 0

        trial_results_dto['reward_at_success'] = total_reward if tiles == tiles_to_complete else reward_at_success
        trial_results_dto['reward_at_optimum_steps'] = total_reward if steps_ran == optimal_steps else reward_at_optimum_steps

    def _print_trial_results(self, trial, elapsed_time, total_reward, steps_ran, trial_results_dto):
        test_name = trial_results_dto['test_name']
        optimal_steps = trial_results_dto['optimal_steps']
        optimal_reward = trial_results_dto['optimal_reward']
        random_reward = trial_results_dto['random_reward']
        test_success = trial_results_dto['test_success']
        steps_at_success = trial_results_dto['steps_at_success']
        reward_at_optimum_steps = round(trial_results_dto['reward_at_optimum_steps'], 2)
        reward_at_success = round(trial_results_dto['reward_at_success'], 2)

        optimal_reward_diff = self._reward_diff_percentage(reward_at_optimum_steps, optimal_reward)
        random_reward_diff = self._reward_diff_percentage(reward_at_optimum_steps, random_reward)

        trial_str = '' if trial is None else f'Planning trial: {trial}'
        elapsed_time_str = '' if elapsed_time is None else f'Elapsed_time: {round(elapsed_time,0)}'
        success_str = f'Test success: {test_success} | Reward at success: {reward_at_success} | Steps on Success: {steps_at_success}\n' \
            if test_success else f'Test success: {test_success}\n'
        print(
            f'\n\nRESULTS FOR: {test_name}\n{trial_str} | {elapsed_time_str}\n'
            f'Total reward: {round(total_reward, 2)} | Max reward: {trial_results_dto["max_reward"]} |  Steps on exit: {steps_ran}\n'
            f'{success_str}'
            f'Agent reward at optimal step {optimal_steps} : {reward_at_optimum_steps}\n'
            f'Manual drive reward at  step {optimal_steps} : {optimal_reward} | Reward Diff: {optimal_reward_diff} %\n'
            f'Random drive reward at  step {optimal_steps} : {random_reward}  | Reward Diff: {random_reward_diff} %')


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
        self._print_total_steps(actions)
        print(f"Action sequence [(action, repetition),...]: {actions}")
        for _ in range(self.trials):
            self.environment.reset(seed=self.seed)
            self.simulated_environment.reset()
            self._set_car_pos(start_track)
            init_state = self._skip_zoom()
            total_reward_real, total_partial_reward_sim, avg_recon_diff, total_full_sim_reward = self._step(actions,
                                                                                                            init_state)
            self._print_results(total_reward_real, total_full_sim_reward, total_partial_reward_sim, avg_recon_diff)

    def _reconstruct(self, state):
        state = self.preprocessor.resize_frame(state).unsqueeze(0)
        reconstruction, z_mean, z_log_standard_deviation = self.vae(state)
        latent_state = self.vae.sample_reparametarization(z_mean, z_log_standard_deviation)
        return reconstruction, latent_state

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
                vae_reconstruction, latent_state = self._reconstruct(current_state)
                latent_state, simulated_reward, simulated_is_done, hidden_state = self.simulated_environment.step(
                    action, hidden_state, latent_state,
                    is_simulation_real_environment=True)
                total_reward_real += real_reward
                total_reward_sim += simulated_reward
                self.vae_render.render(vae_reconstruction)
                self.environment.render()

                mdrnn_reconstruction = self.simulated_environment.current_reconstruction
                total_recon_diff += self._compare_reconstructions(vae_reconstruction[0].reshape((64, 64, 3)).numpy(),
                                                                  mdrnn_reconstruction)
                steps += 1

        return total_reward_real, total_reward_sim, total_recon_diff / steps, total_full_sim_reward

    def _step_sequence_in_dream(self, actions, current_state, hidden):
        vae_reconstruction, latent = self._reconstruct(current_state)
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

    def _skip_zoom(self):
        state = None
        for _ in range(50):
            state, _, _, _ = self.environment.step([0, 0, 0])
        self.environment.render()
        return state

    def _compare_reconstructions(self, vae_recon, mdrnn_recon):
        vae_grey = np.mean(vae_recon, axis=2)
        mdrnn_grey = np.mean(mdrnn_recon, axis=2)
        dists = [dist.euclidean(vae_grey[i], mdrnn_grey[i]) for i in range(len(vae_grey))]
        return np.mean(dists)