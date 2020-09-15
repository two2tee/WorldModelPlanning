""" RMHC for simulated environments """
#  Copyright (c) 2020, - All Rights Reserved
#  This file is part of the Evolutionary Planning on a Learned World Model thesis.
#  Unauthorized copying of this file, via any medium is strictly prohibited without the consensus of the authors.
#  Written by Thor V.A.N. Olesen <thorolesen@gmail.com> & Dennis T.T. Nguyen <dennisnguyen3000@yahoo.dk>.

import copy
import torch
from concurrent.futures import as_completed
from planning.interfaces.individual import Individual
from tuning.evolution_handler import EvolutionHandler
from concurrent.futures.thread import ThreadPoolExecutor
from utility.logging.single_step_logger import SingleStepLogger
from planning.interfaces.abstract_hill_climb_simulation import AbstractRandomMutationHillClimbing


class RMHC(AbstractRandomMutationHillClimbing):
    def __init__(self, horizon, max_generations, is_shift_buffer, is_rollout, max_rollouts=None, rollout_length=None,
                 is_parallel_rollouts=False, is_delta_sampling=False):
        super().__init__(horizon, max_generations, is_shift_buffer, is_rollout, max_rollouts, rollout_length)
        self.current_elite = None
        self.latent = None
        self.hidden = None
        self.elite_history = []
        self.is_parallel_rollouts = is_parallel_rollouts
        self.is_delta_sampling = is_delta_sampling

        self.evolution_handler = EvolutionHandler(self.horizon)
        self.mutation_operator = self.evolution_handler.get_mutation_operator()

    def search(self, environment, latent, hidden):
        self.latent = latent
        self.hidden = hidden
        self.elite_history = []
        self.current_elite = self._initialize_individual(environment)

        self._evaluate_individual(self.current_elite, environment)
        self._append_elite(self.current_elite)

        # logger = SingleStepLogger(is_logging=True) # TODO REMOVE
        # logger.start_log(f'World_Model_RandomNormal_RMHC_h{self.horizon}_g{self.max_generations}')
        for generation in range(self.max_generations):
            mutated_individual = self._mutate(environment, self.current_elite, generation)
            self.current_elite = self._select_best_individual(self.current_elite, mutated_individual, environment)

            # logger.log_acc_reward_single_planning_step(test_name='planning_head_to_grass_right', step=generation, acc_reward=self.current_elite.fitness, actions=self.current_elite.action_sequence)
        # logger.end_log()

        best_action = self.current_elite.action_sequence[0]
        return best_action, self.elite_history

    def _initialize_individual(self, environment):
        if self.is_shift_buffer and self.current_elite is not None:
            individual = self._shift_buffer(environment, self.current_elite)
        else:
            previous_action = None
            action_sequence = []
            for _ in range(self.horizon):
                action_sequence.append(environment.sample(previous_action) if self.is_delta_sampling else environment.sample())
            individual = Individual(action_sequence)
        individual.fitness, individual.age = 0, 0  # reset across generations
        return individual

    def _select_best_individual(self, current_elite, mutated_individual, simulated_environment):
        self._evaluate_individual(mutated_individual, simulated_environment)
        elite = mutated_individual if mutated_individual.fitness > current_elite.fitness else current_elite
        self._append_elite(elite)
        return elite

    def _shift_buffer(self, environment, individual):
        individual.action_sequence.pop(0)
        individual.action_sequence.append(environment.sample(individual.action_sequence[-1]) if self.is_delta_sampling else environment.sample())
        return individual

    def _rollout(self, environment, latent, hidden, is_parallel=True):
        total_reward = 0
        if is_parallel:
            with ThreadPoolExecutor() as executor:
                rollout_futures = [executor.submit(lambda args: self._single_rollout(*args), [environment, latent, hidden]) for _ in range(self.max_rollouts)]
                total_reward += sum([rollout_future.result() for rollout_future in as_completed(rollout_futures)])
        else:
            total_reward += sum([self._single_rollout(environment, latent, hidden) for _ in range(self.max_rollouts)])

        return total_reward / self.max_rollouts

    def _single_rollout(self, environment, latent, hidden):
        is_done = False
        total_reward = 0
        rollout_step = 0
        rollout_latent = latent
        rollout_hidden = hidden

        while not is_done and rollout_step < self.rollout_length:
            action = environment.sample()
            rollout_latent, reward, is_done, rollout_hidden = environment.step(action, rollout_hidden, rollout_latent,
                                                                               is_simulation_real_environment=False)
            total_reward += reward
            rollout_step += 1
        return total_reward

    def _mutate(self, environment, current_elite, generation):
        individual = copy.deepcopy(current_elite)
        self.mutation_operator(environment, individual)
        individual.age, individual.fitness = generation + 1, 0
        return individual

    def _evaluate_individual(self, individual, environment):
        with torch.no_grad():
            is_done = False
            total_reward = 0
            latent = self.latent
            hidden = self.hidden

            for action in individual.action_sequence:
                if not is_done:
                    latent, reward, is_done, hidden = environment.step(action, hidden, latent, is_simulation_real_environment=False)
                    total_reward += reward
                else:
                    break

            if self.is_rollout and not is_done:
                total_reward += self._rollout(environment, latent, hidden, self.is_parallel_rollouts)
            individual.fitness += total_reward

    def _append_elite(self, individual):
        is_new_elite = len(self.elite_history) is 0 or individual.age is not self.current_elite.age
        self.elite_history.append((individual.fitness, is_new_elite, individual.action_sequence))
