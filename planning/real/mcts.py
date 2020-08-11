""" MCTS algorithm for non-simulated environments """
#  Copyright (c) 2020, - All Rights Reserved
#  This file is part of the Evolutionary Planning on a Learned World Model thesis.
#  Unauthorized copying of this file, via any medium is strictly prohibited without the consensus of the authors.
#  Written by Thor V.A.N. Olesen <thorolesen@gmail.com> & Dennis T.T. Nguyen <dennisnguyen3000@yahoo.dk>.

import gym
import copy
import math
import random
from planning.interfaces.node import Node
from planning.interfaces.abstract_tree_search import AbstractTreeSearch


def uct(node, temperature=None):  # Upper Confidence Bound for Trees
    exploit_average_reward = node.total_reward / node.visit_count  # Q(s,a)
    exploration = math.sqrt(math.log(node.parent.visit_count) / node.visit_count)
    return exploit_average_reward + temperature * exploration


class MCTS(AbstractTreeSearch):

    def __init__(self, temperature, max_rollouts, rollout_length=None):
        super().__init__(temperature, max_rollouts, rollout_length)
        self.root = None

    def search(self, environment):
        self.root = Node(actions=action_space(environment)) if self.root is None else self.root  # Reuse tree

        for _ in range(self.max_rollouts):
            total_reward = 0
            current_environment = copy.deepcopy(environment)
            selection_reward, node, is_done = self.selection(self.root, current_environment)
            total_reward += selection_reward

            if not is_done:
                expansion_reward, node, is_done = self.expansion(node, current_environment)
                simulation_reward = self.simulation(current_environment, is_done)
                total_reward += expansion_reward + simulation_reward

            self.backpropagation(node, total_reward)

        best_child = self.select_best_child(self.root, temperature=0)

        self.root = best_child
        self.root.parent = None

        return best_child.action

    def selection(self, node, environment):
        is_done = False
        selection_reward = 0
        while node.children:
            if node.is_fully_expanded():
                node = self.select_best_child(node)
                _, reward, is_done, _ = environment.step(node.action)
                selection_reward += reward
            else:
                break
        return selection_reward, node, is_done

    def expansion(self, node, environment):
        random_index = random.randrange(len(node.actions))
        random_action = node.actions.pop(random_index)
        _, reward, is_done, _ = environment.step(random_action)
        actions = action_space(environment)
        child_node = Node(parent=node, action=random_action, actions=actions)
        node.children.append(child_node)
        return reward, child_node, is_done

    def simulation(self, environment, is_done):
        simulation_reward = 0
        while not is_done:
            random_action = sample(environment)
            _, reward, is_done, _ = environment.step(random_action)
            simulation_reward += reward
        return simulation_reward

    def backpropagation(self, node, total_reward):
        while node:
            node.visit_count += 1
            node.total_reward += total_reward
            node = node.parent

    def select_best_child(self, node, temperature=None, selection_criteria=uct):
        temperature = temperature if temperature is not None else self.temperature
        return max(node.children, key=lambda child: selection_criteria(child, temperature))


def action_space(environment, delta=1):
    gym.logger.set_level(40)
    if isinstance(environment.action_space, gym.spaces.Discrete):
        return list(range(environment.action_space.n))
    else:
        import numpy as np
        start = environment.action_space.low[0]
        stop = environment.action_space.high[0]  # exlusive so add 0.1
        actions = [[round(a, 1)] for a in np.arange(start, stop + 0.1, delta)]
        return actions


def sample(environment):
    gym.logger.set_level(40)
    if isinstance(environment.action_space, gym.spaces.Discrete):
        return random.choice(action_space(environment))
    else:
        return [random.uniform(environment.action_space.low[0], environment.action_space.high[0])]
