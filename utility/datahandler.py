""" DataHandler to generate rollouts - Random and Good policy are usable
    The good policy is based on https://worldmodels.github.io
"""
#  Copyright (c) 2020, - All Rights Reserved
#  This file is part of the Evolutionary Planning on a Learned World Model thesis.
#  Unauthorized copying of this file, via any medium is strictly prohibited without the consensus of the authors.
#  Written by Thor V.A.N. Olesen <thorolesen@gmail.com> & Dennis T.T. Nguyen <dennisnguyen3000@yahoo.dk>.

import os
import math
import random
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from PIL import Image
from multiprocessing import Pool
from os.path import exists, join
from .carracing.model import Model
from gym.envs.box2d.car_dynamics import Car


class DataHandler:
    def __init__(self, config):
        self.data_dir = config["data_generator"]['data_output_dir']
        if not exists(self.data_dir):
            os.mkdir(self.data_dir)
        self.config = config
        self.is_corners_only = config["data_generator"]['is_corners_only']
        self.rollouts = config["data_generator"]['rollouts']
        self.sequence_length = config["data_generator"]['sequence_length']
        self.threads = multiprocessing.cpu_count()
        self.render_mode = False
        self.img_width = 64
        self.img_height = 64
        print(f'data_dir: {self.data_dir} | cores: {self.threads}')

    def generate_rollouts(self, game):
        self.threads = self.rollouts if self.rollouts < self.threads else self.threads
        rollouts_per_thread = int(self.rollouts / self.threads)
        print(f'{self.rollouts} rollouts across {self.threads} cores - {rollouts_per_thread} rollouts per thread.')
        # multiprocessing.set_start_method('spawn')  # spawn only works over fork on Mac
        with Pool(self.threads) as pool:
            threads = [pool.apply_async(self._run_rollout_thread, args=(rollouts_per_thread, thread))
                       for thread in range(1, self.threads + 1)]
            [thread.get() for thread in threads]
            pool.close()

        print(f'Done - {self.rollouts} rollout samples for {game} saved in {self.data_dir}')

    def _run_rollout_thread(self, rollouts, thread):
        model = Model(load_model=True)
        model.make_env(self.render_mode)
        model.load_model(f"{os.getcwd()}/utility/carracing/log/carracing.cma.16.64.best.json")

        i = 1
        while i < rollouts + 1:
            actions_rollout, states_rollout, reward_rollout, is_done_rollout = self._corners_rollout(model) if self.is_corners_only else \
                                                                               self._standard_rollout(model)

            if len(actions_rollout) < 128:  # ensure rollouts contains enough data for sequence
                print(f'thread: {thread} - Bad rollout with {len(actions_rollout)} actions - retry...')
                i -= 1
            else:
                self._save_rollout(thread, i, states_rollout, reward_rollout, actions_rollout, is_done_rollout)
            i += 1

        return thread

    def _corners_rollout(self, model):
        obs, model, car_position = self._reset(model)
        actions_rollout, states_rollout, reward_rollout, is_done_rollout = [], [], [], []
        action = [0, 0, 0]
        steps = 0
        steps_since_exit = 0
        corners = 0
        is_corner = False
        is_in_corner_pass_checkpoint = False
        is_out_corner_pass_checkpoint = True

        plt.figure()
        resized = (64, 64, 3)
        monitor = plt.imshow(X=np.zeros(resized, dtype=np.uint8))

        for i in range(car_position, len(model.env.track)):
            model.env.car = Car(model.env.world, *model.env.track[i][1:4])

            frame, reward, done, _ = model.env.step([0, 0, 0])
            frame = self._compress_frame(frame, True)
            model.env.render()

            if self._is_corner(frame, is_in=True):
                for _ in range(10):
                    frame, reward, done, _ = model.env.step([0, 0, 0])
                break

        for t in range(self.sequence_length):
            obs, reward, done, info, action = self._step(model, obs, action, is_corner)
            obs = self._compress_frame(obs, is_resize=True)

            # Check for corner entry
            if is_in_corner_pass_checkpoint and self._is_corner(obs, is_in=True):
                is_corner = True
                is_in_corner_pass_checkpoint = False
                corners += 1
                steps_since_exit = 0
            else:
                is_in_corner_pass_checkpoint = self._corner_in_checkpoint(obs)

            if is_corner:
                # Capture data on corner
                model.env.viewer.window.dispatch_events()
                actions_rollout.append(action)
                states_rollout.append(obs)
                reward_rollout.append(reward)
                is_done_rollout.append(done)

                model.env.render()
                monitor.set_data(obs)
                plt.pause(0.1)

                # Check for corner exit
                if corners > 0:
                    if self._is_corner(obs, is_in=False):
                        is_out_corner_pass_checkpoint = False
                    elif not is_out_corner_pass_checkpoint:
                        is_out_corner_pass_checkpoint = self._corner_out_checkpoint(obs)
                        if is_out_corner_pass_checkpoint:
                            corners -= 1

                if corners == 0 and steps >= 128 and steps_since_exit > 20 or done:
                    break
                steps_since_exit = steps_since_exit + 1 if corners == 0 else steps_since_exit
                steps += 1
            if not is_corner and t >= int(self.sequence_length/2):  # Early stop if corner not reached
                break

        return actions_rollout, states_rollout, reward_rollout, is_done_rollout

    def _standard_rollout(self, model):
        action = [0, 0, 0]
        obs, model, _ = self._reset(model)
        actions_rollout, states_rollout, reward_rollout, is_done_rollout = [], [], [], []
        for t in range(self.sequence_length):
            obs, reward, done, info, action = self._step(model, obs, action)
            obs = self._compress_frame(obs, is_resize=True)
            model.env.viewer.window.dispatch_events()
            actions_rollout.append(action)
            states_rollout.append(obs)
            reward_rollout.append(reward)
            is_done_rollout.append(done)
            if done:
                break
        return actions_rollout, states_rollout, reward_rollout, is_done_rollout

    def _step(self, model, obs, previous_action, is_corner=False):
        if self.config["data_generator"]["is_ha_agent_driver"] or (not is_corner and self.is_corners_only):
            model.env.render("human") if self.render_mode else model.env.render('rgb_array')
            z, mu, logvar = model.encode_obs(obs)
            action = model.get_action(z)
        else:
            action = self.brownian_sample(previous_action)
        obs, reward, done, info = model.env.step(action)

        return obs, reward, done, info, action

    def _reset(self, model):
        # Generate random tracks and initial car positions
        seed = random.randint(0, 2 ** 31 - 1)
        model.env.seed(seed)
        model.reset()
        model.env.reset()
        obs = [model.env.step([0, 0, 0])[0] for _ in range(50)][-1]

        # Worse total reward by randomizing car position
        car_position = np.random.randint(len(model.env.track))
        model.env.car = Car(model.env.world, *model.env.track[car_position][1:4])

        # Garbage collection of events in viewer
        model.env.viewer.window.dispatch_events()
        return obs, model, car_position

    def _compress_frame(self, frame, is_resize=False):
        if is_resize:
            frame = np.array(Image.fromarray(frame).resize(size=(self.img_width, self.img_height)))
        return frame

    def _corner_in_checkpoint(self, frame):
        return not self._is_corner(frame, is_in=True)

    def _corner_out_checkpoint(self, frame):
        return not self._is_corner(frame, is_in=False)

    def _is_corner(self, frame, is_in):
        corner_type = self.config['data_generator']['corner_type']  # all, left, right

        y = 0 if is_in else 55  # top else bot
        pixel_row = frame[y][:][:]

        for i, pixel in enumerate(pixel_row):
            if self._is_white_pixel(pixel) or self._is_red_pixel(pixel):
                previous_pixel = pixel_row[i-1]
                if corner_type == 'right' and self._is_grey_pixel(previous_pixel):
                    return True
                if corner_type == 'left' and not self._is_grey_pixel(previous_pixel):
                    return True
                if corner_type == 'all':
                    return True
        return False

    def _is_grey_pixel(self, pixel):
        return pixel[0] < 110 and pixel[1] < 110 and pixel[2] < 110

    def _is_white_pixel(self, pixel):
        return pixel[0] > 200 and pixel[1] > 200 and pixel[2] > 200

    def _is_red_pixel(self, pixel):
        return pixel[0] > 150 and pixel[1] < 200 and pixel[2] < 200

    def _save_rollout(self, thread, rollout_number, states_rollout, reward_rollout, actions_rollout, is_done_rollout):
        print(f"Thread {thread} - End of rollout {rollout_number}, {len(states_rollout)} frames.")
        print(self.data_dir, f'thread_{thread}_rollout_{rollout_number}')
        np.savez_compressed(file=join(self.data_dir, f'{self.config["data_generator"]["data_prefix"]}thread_{thread}_resized_rollout_{rollout_number}'),
                            observations=np.array(states_rollout),
                            rewards=np.array(reward_rollout),
                            actions=np.array(actions_rollout),
                            terminals=np.array(is_done_rollout))

    def brownian_sample(self, previous_action, delta = 1. / 50):  # a_{t+1} = a_t + sqrt(dt) N(0, 1)
        dactions_dt = np.random.randn(len(previous_action))
        new_action = [0, 0, 0]
        new_action[0] = np.clip(previous_action[0] + math.sqrt(delta) * dactions_dt[0], -1, 1)
        new_action[1] = np.clip(previous_action[1] + math.sqrt(delta) * dactions_dt[1], 0, 1)
        new_action[2] = np.clip(previous_action[2] + math.sqrt(delta) * dactions_dt[2], 0, 1)
        return new_action
