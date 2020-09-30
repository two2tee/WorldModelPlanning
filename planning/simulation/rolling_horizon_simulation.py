""" RHEA for simulated environments """
#  Copyright (c) 2020, - All Rights Reserved
#  This file is part of the Evolutionary Planning on a Learned World Model thesis.
#  Unauthorized copying of this file, via any medium is strictly prohibited without the consensus of the authors.
#  Written by Thor V.A.N. Olesen <thorolesen@gmail.com> & Dennis T.T. Nguyen <dennisnguyen3000@yahoo.dk>.

import copy
import torch
from planning.interfaces.individual import Individual
from tuning.evolution_handler import EvolutionHandler
from concurrent.futures import ThreadPoolExecutor, as_completed
from planning.interfaces.abstract_rolling_horizon_simulation import AbstractRollingHorizon


class RHEA(AbstractRollingHorizon):
    def __init__(self, population_size, horizon, max_generations, is_shift_buffer, is_rollout, max_rollouts=None, rollout_length=None,
                 is_parallel_rollouts=False):
        super().__init__(population_size, horizon, max_generations, is_shift_buffer, is_rollout, max_rollouts, rollout_length)
        print(self.is_rollout)
        print(self.max_rollouts)
        self.latent = None
        self.hidden = None
        self.population = None
        self.elite_history = []
        self.current_elite = None
        self.is_parallel_rollouts = is_parallel_rollouts

        self.evolution_handler = EvolutionHandler(self.horizon)
        self.selection_type = self.evolution_handler.get_selection_type()
        self.genetic_operator = self.evolution_handler.get_genetic_operator()
        self.mutation_operator = self.evolution_handler.get_mutation_operator()
        self.crossover_operator = self.evolution_handler.get_crossover_operator()

    def search(self, environment, latent, hidden):
        self.latent = latent
        self.hidden = hidden
        self.elite_history = []
        self.current_elite = None
        self.population = self.initialize_population(environment, self.population_size)

        for generation in range(self.max_generations):
            self.evaluate_population(self.population, environment)
            self.current_elite = self._elitist_selection(self.population)
            self.population = self.evolve_population(environment, generation, self.population)

        self.evaluate_population(self.population, environment)
        self.current_elite = self._elitist_selection(self.population)
        best_action = self.current_elite.action_sequence[0]
        return best_action, self.elite_history

    def _reset_i(self, i):
        if i != self.current_elite:
            i.age += 1
            i.fitness = 0
        return i

    def initialize_population(self, environment, population_size):
        if self.population and self.is_shift_buffer:
            return list(self.shift_buffer(environment, self.population))
        else:
            return [self._generate_individual(environment, self.horizon) for _ in range(population_size)]

    def shift_buffer(self, environment, population):
        for individual in population:
            individual.age = 0
            individual.fitness = 0
            individual.action_sequence.pop(0)
            individual.action_sequence.append(environment.sample())
            yield individual

    def _generate_individual(self, environment, horizon):
        previous_action = None
        action_sequence = []
        for _ in range(horizon):
            action_sequence.append(environment.sample())
        return Individual(action_sequence)

    def evaluate_population(self, population, environment, is_parallel=True):
        population = population if self.current_elite is None else [individual for individual in population
                                                                    if individual is not self.current_elite]
        if is_parallel:
            with ThreadPoolExecutor() as executor:
                processes = [executor.submit(lambda args: self._evaluate_individual(*args), [individual, environment])
                             for individual in population]
                [task.result() for task in as_completed(processes)]  # non-blocking async
        else:
            [self._evaluate_individual(individual, environment) for individual in population]

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

    def _rollout(self, environment, latent, hidden, is_parallel=False):
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

    def evolve_population(self, environment, generation, population):
        next_population = [self.current_elite]
        for _ in range(self.population_size - 1):
            parent_a, parent_b = self.selection_type(population)
            child = self.variation(environment, parent_a, parent_b, generation)
            next_population.append(child)
        return next_population

    def variation(self, environment, parent_a, parent_b, generation):
        child = None
        if self.genetic_operator == 'cross' or self.genetic_operator == 'crossmut':
            child = self.crossover_operator(parent_a, parent_b, generation)

            if self.genetic_operator == 'crossmut':  # if crossover + mutation
                self.mutation_operator(environment, child)

        if self.genetic_operator == 'mut':  # if mutation only
            best_sequence = parent_a.action_sequence if parent_a.fitness > parent_b.fitness else parent_b.action_sequence
            child = Individual(copy.deepcopy(best_sequence), age=generation)
            self.mutation_operator(environment, child)

        child.age += 1
        return child

    def _elitist_selection(self, population):
        population = sorted(population, key=lambda individual: individual.fitness, reverse=True)
        elite = population[0]
        self._append_elite(elite)
        return elite

    def _append_elite(self, individual):
        is_new_elite = len(self.elite_history) is 0 or individual.age is not self.current_elite.age
        self.elite_history.append((individual.fitness, is_new_elite, individual.action_sequence))
