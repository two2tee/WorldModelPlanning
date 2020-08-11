""" RHEA for simulated environments """
#  Copyright (c) 2020, - All Rights Reserved
#  This file is part of the Evolutionary Planning on a Learned World Model thesis.
#  Unauthorized copying of this file, via any medium is strictly prohibited without the consensus of the authors.
#  Written by Thor V.A.N. Olesen <thorolesen@gmail.com> & Dennis T.T. Nguyen <dennisnguyen3000@yahoo.dk>.

import math
import random
from planning.interfaces.node import Node
from planning.interfaces.abstract_tree_search_simulation import AbstractTreeSearch


def uct(node, temperature=None):  # Upper Confidence Bound for Trees
    exploit_average_reward = node.total_reward / node.visit_count  # Q(s,a)
    exploration = math.sqrt(math.log(node.parent.visit_count) / node.visit_count)
    return exploit_average_reward + temperature * exploration


class MCTS(AbstractTreeSearch):

    def __init__(self, temperature, max_rollouts, rollout_length, is_discrete_delta):
        super().__init__(temperature, max_rollouts, rollout_length)
        self.is_discrete_delta = is_discrete_delta
        self.latent = None
        self.hidden = None

    def search(self, environment, latent, hidden):
        action = [0, 0, 0] if self.root is None else self.root.action
        self.root = Node(action=action, actions=environment.discrete_action_space(action)) if self.root is None else self.root  # Reuse tree

        for _ in range(self.max_rollouts):
            total_reward = 0
            self.latent = latent
            self.hidden = hidden

            selection_reward, node, is_done = self._selection(self.root, environment)
            total_reward += selection_reward

            if not is_done:
                expansion_reward, node, is_done = self._expansion(node, environment)
                simulation_reward = self._simulation(environment, is_done)
                total_reward += expansion_reward + simulation_reward

            self._backpropagation(node, total_reward)

        best_child = self._select_best_child(self.root, temperature=0)

        self.root = best_child
        self.root.parent = None

        return best_child.action

    def _selection(self, node, environment):
        is_done = False
        selection_reward = 0
        while node.children:
            if node.is_fully_expanded():
                node = self._select_best_child(node)
                self.latent, reward, is_done, self.hidden = environment.step(node.action, self.hidden, self.latent,
                                                                             is_simulation_real_environment=False)
                selection_reward += reward
            else:
                break

        return selection_reward, node, is_done

    def _expansion(self, node, environment):
        random_index = random.randrange(len(node.actions))
        random_action = node.actions.pop(random_index)
        self.latent, reward, is_done, self.hidden = environment.step(node.action, self.hidden, self.latent,
                                                                     is_simulation_real_environment=False)
        actions = environment.discrete_action_space(random_action)
        child_node = Node(parent=node, action=random_action, actions=actions)
        node.children.append(child_node)
        return reward, child_node, is_done

    def _simulation(self, environment, is_done):
        simulation_reward = 0
        simulation_counter = 0
        random_action = None
        while not is_done and simulation_counter < self.rollout_length:
            random_action = environment.discrete_delta_sample(random_action) if self.is_discrete_delta else environment.discrete_sample()
            self.latent, reward, is_done, self.hidden = environment.step(random_action, self.hidden, self.latent,
                                                                         is_simulation_real_environment=False)
            simulation_reward += reward
            simulation_counter += 1
        return simulation_reward

    def _backpropagation(self, node, total_reward):
        while node:
            node.visit_count += 1
            node.total_reward += total_reward
            node = node.parent

    def _select_best_child(self, node, temperature=None, selection_criteria=uct):
        temperature = temperature if temperature is not None else self.temperature
        return max(node.children, key=lambda child: selection_criteria(child, temperature))
