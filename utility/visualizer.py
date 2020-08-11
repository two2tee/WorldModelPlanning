""" Tool to visualize fitness plots and car trajectories """
#  Copyright (c) 2020, - All Rights Reserved
#  This file is part of the Evolutionary Planning on a Learned World Model thesis.
#  Unauthorized copying of this file, via any medium is strictly prohibited without the consensus of the authors.
#  Written by Thor V.A.N. Olesen <thorolesen@gmail.com> & Dennis T.T. Nguyen <dennisnguyen3000@yahoo.dk>.

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from utility.trajectories import get_trajectory

fitness_figure = plt.figure(num=random.randint(0, 999))
trajectory_figure = plt.figure(num=random.randint(0, 999))
fitness_figure_num = fitness_figure.number
trajectory_figure_num = trajectory_figure.number


class Visualizer:

    def show_fitness_plot(self, max_generation, elites, agent=None):
        plt.figure(num=fitness_figure_num)
        plt.clf()
        x = np.arange(start=1, stop=len(elites) + 1)
        y = [fitness[0] for fitness in elites]

        plt.plot(x, y)
        is_same_labelled, is_new_labelled = False, False
        for i, data in enumerate(elites):
            i = i+1
            if data[1]:  # Is elite
                plt.scatter(i, data[0], c='red', label='New elite') if not is_new_labelled else plt.scatter(i, data[0], c='red')
                is_new_labelled = True
            else:
                plt.scatter(i, data[0], c='blue', label='Same elite') if not is_same_labelled else plt.scatter(i, data[0], c='blue')
                is_same_labelled = True

        plt.title(f'{"" if agent is None else agent+" - "}Fitness of elite for {max_generation} generations')
        plt.xticks(x)
        plt.xlabel('Generations')
        plt.ylabel('Fitness')
        plt.legend(loc='lower right')
        plt.grid()
        plt.pause(.01)

    def show_trajectory_plot(self, frame, elites, agent=None, env=None, is_render_best_elite_only=True):

        plt.figure(trajectory_figure_num)
        plt.clf()
        plt.imshow(X=frame)

        elites_extracted = []

        for i, data in enumerate(elites):
            elite_fitness, is_new_elite, action_sequence = data
            if is_new_elite:
                elites_extracted.append((i+1, elite_fitness, get_action_sequence(action_sequence)))

        # ALL PREVIOUS ELITE TRAJECTORIES
        if not is_render_best_elite_only:
            for i in range(len(elites_extracted)-1):
                gen, elite_fitness, action_sequence = elites_extracted[i]
                label = f"Elite at gen {gen} - Fitness {round(elite_fitness, 4)}"
                forward_velocities = get_trajectory(env, action_sequence, is_best_elite=False)
                self._set_trajectories(action_sequence, forward_velocities, label, is_best_elite=False)

        # BEST ELITE TRAJECTORY
        gen, elite_fitness, action_sequence = elites_extracted[len(elites_extracted)-1]  # Best elite
        label = f"BEST Elite at gen {gen} - Fitness {round(elite_fitness, 4)}"
        forward_velocities = get_trajectory(env, action_sequence, is_best_elite=True)
        self._set_trajectories(action_sequence, forward_velocities, label, is_best_elite=True)


        applied_action = (np.around(action_sequence[0], 2))
        plt.title(f'{"" if agent is None else agent+" - "} Planning Trajectories'
                  f'\nGas: {applied_action[1]}'
                  f'\nSteer: {applied_action[0]}'
                  f'\nBrake: {applied_action[2]}')

        plt.legend()
        plt.pause(.01)

    def _set_trajectories(self, action_sequence, forward_velocities, label, is_best_elite):
        x, y = [96/2], [65]
        car_angle = 90
        steer_scaling = 15  # 9

        for i, action in enumerate(action_sequence):
            next_x, next_y = forward_velocities[i]  # Normalized
            next_x = next_x / math.sqrt(next_x ** 2 + next_y ** 2)
            next_y = next_y / math.sqrt(next_x ** 2 + next_y ** 2)
            speed = np.sqrt(np.square(next_x) + np.square(next_y))

            steer_angle = action[0] * steer_scaling

            if speed > 0:
                if car_angle == 0 and steer_angle > 0:
                    car_angle = 359
                if car_angle == 359 and steer_angle < 0:
                    car_angle = 0

                if steer_angle < 0:  # Turn left
                    car_angle = car_angle + (-steer_angle)
                elif steer_angle > 0:  # Turn right
                    car_angle = car_angle - steer_angle

            x_movement = math.cos(math.radians(car_angle))
            y_movement = math.sin(math.radians(car_angle))

            trajectory_x = x[-1] + (x_movement * next_x)
            trajectory_y = y[-1] - (y_movement * next_y)

            x.append(trajectory_x)
            y.append(trajectory_y)

        plt.plot(x, y, lw=1, label=label) if not is_best_elite else plt.plot(x, y, lw=3, label=label, c='red')


def get_action_sequence(actions):
    action = actions[0]
    if type(action) == tuple and len(action) == 2:
        return [action for (action, _) in actions]
    return actions

