import os
import time
import torch
import pickle
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from pprint import pprint
from datetime import date
from os.path import join, exists

from tests.base_planning_tester import BasePlanningTester, TEST_NAME, ELITES, CUSTOM_SEED, ACTION_HISTORY
from utility.visualizer import Visualizer
from gym.envs.box2d.car_dynamics import Car
from concurrent.futures import ProcessPoolExecutor

# custom args
OPTIMAL_STEPS = 'optimal_steps'
OPTIMAL_REWARD = 'optimal_reward'
RANDOM_REWARD = 'random_reward'
TILES_TO_COMPLETE = 'tiles_to_complete'


class PlanningTester(BasePlanningTester):
    def __init__(self, config, vae, mdrnn, preprocessor, environment, planning_agent):
        super().__init__(config, vae, mdrnn, preprocessor, environment, planning_agent)
        self.car_forward_velocity_approx = 0
        self.velocity_growth_penalizer = 0.005
        self.visualizer = Visualizer()
        self.is_render_best_elite_only = self.config['visualization']['is_render_best_elite_only']
        self.is_render_fitness = self.config['visualization']['is_render_fitness']
        self.is_render_trajectory = self.config['visualization']['is_render_trajectory']


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

    def _step_sequence_in_dream(self, actions, current_state, hidden):
        latent, _ = self._encode_state(current_state)
        total_reward = 0

        for action in actions:
            latent, simulated_reward, simulated_is_done, hidden = self.simulated_environment.step(
                self._get_action(action), hidden, latent, is_simulation_real_environment=True)
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
            OPTIMAL_STEPS: args[OPTIMAL_STEPS],
            OPTIMAL_REWARD: args[OPTIMAL_REWARD],
            RANDOM_REWARD: args[RANDOM_REWARD],
            'tiles_to_complete': args[TILES_TO_COMPLETE]
        }

    def _update_trial_results(self, trial_results_dto, reward, total_reward, steps_ran):
        tiles = trial_results_dto['tiles']
        tiles_to_complete = trial_results_dto['tiles_to_complete']
        reward_at_success = trial_results_dto['reward_at_success']
        reward_at_optimum_steps = trial_results_dto['reward_at_optimum_steps']
        optimal_steps = trial_results_dto[OPTIMAL_STEPS]

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
        optimal_reward = trial_results_dto[OPTIMAL_REWARD]
        random_reward = trial_results_dto[RANDOM_REWARD]
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
