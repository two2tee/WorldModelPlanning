""" RHEA for non-simulated environments """
#  Copyright (c) 2020, - All Rights Reserved
#  This file is part of the Evolutionary Planning on a Learned World Model thesis.
#  Unauthorized copying of this file, via any medium is strictly prohibited without the consensus of the authors.
#  Written by Thor V.A.N. Olesen <thorolesen@gmail.com> & Dennis T.T. Nguyen <dennisnguyen3000@yahoo.dk>.

import gym
import copy
import random
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from planning.interfaces.individual import Individual
from planning.interfaces.abstract_rolling_horizon import AbstractRollingHorizon


class RHEA(AbstractRollingHorizon):
    def __init__(self, population_size, horizon, max_generations, is_shift_buffer, is_rollout, mutation_probability=None):
        super().__init__(population_size, horizon, max_generations, is_shift_buffer, is_rollout, mutation_probability)
        self.population = None
        self.current_elite = None

    def search(self, environment):

        self.current_elite = None
        self.population = self.initialize_population(environment, self.population_size)

        for generation in range(self.max_generations):
            self.evaluate_population(self.population, environment)
            self.current_elite = self._elitist_selection(self.population)
            self.population = self.evolve_population(environment, generation, self.population)

        self.evaluate_population(self.population, environment)
        self.current_elite = self._elitist_selection(self.population)
        return self.current_elite.action_sequence[0]

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
            individual.action_sequence.append(sample(environment))
            yield individual

    def _generate_individual(self, environment, horizon):
        action_sequence = [sample(environment) for _ in range(horizon)]
        return Individual(action_sequence)

    def evaluate_population(self, population, environment, is_parallel=True):
        population = population if self.current_elite is None else [individual for individual in population
                                                                    if individual is not self.current_elite]

        if is_parallel:
            with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                processes = [executor.submit(lambda args: self._evaluate_individual(*args), [individual, environment])
                             for individual in population]
                [task.result() for task in processes]
        else:
            [self._evaluate_individual(individual, environment) for individual in population]

    def _evaluate_individual(self, individual, environment):
        is_done = False
        total_reward = 0
        environment = copy.deepcopy(environment)

        for action in individual.action_sequence:
            if not is_done:
                _, reward, is_done, _ = environment.step(action)
                total_reward += reward
            else:
                break

        if self.is_rollout and not is_done:
            total_reward += self._rollout(environment, is_done)

        individual.fitness += total_reward

    def _rollout(self, environment, is_done):
        total_reward = 0
        while not is_done:
            action = sample(environment)
            _, reward, is_done, _ = environment.step(action)
            total_reward += reward

        return total_reward

    def evolve_population(self, environment, generation, population):
        next_population = [self.current_elite]
        for _ in range(self.population_size - 1):
            parent_a, parent_b = self.select_parents(population)
            child = self.variation(environment, parent_a, parent_b, generation)
            next_population.append(child)
        return next_population

    def select_parents(self, population, tournament_percentage=0.5):
        tournament_population = random.sample(population, k=int(len(population) * tournament_percentage))  # w/o replace
        population_sorted_by_fitness = sorted(tournament_population, key=lambda individual: individual.fitness, reverse=True)
        return population_sorted_by_fitness[0], population_sorted_by_fitness[1]

    def _cross_over(self, parent_a, parent_b, generation):
        action_sequence = [parent_a.action_sequence[i] if random.uniform(0, 1) < 0.5 else
                           parent_b.action_sequence[i] for i in range(self.horizon)]
        return Individual(action_sequence, age=generation)

    def _mutate(self, environment, child):
        mutation_indices = {random.randrange(self.horizon)}  # At least one gene mutation
        other_mutation_indices = set([i for i, _ in enumerate(child.action_sequence)
                                     if random.uniform(0, 1) < self.mutation_probability])
        mutation_indices = mutation_indices.union(other_mutation_indices)
        mutations = [sample(environment) for _ in range(len(mutation_indices))]
        for i in mutation_indices:
            child.action_sequence[i] = mutations.pop(0)

    def variation(self, environment, parent_a, parent_b, generation):
        child = self._cross_over(parent_a, parent_b, generation)
        self._mutate(environment, child)
        child.age += 1
        return child

    def _elitist_selection(self, population):
        population = sorted(population, key=lambda individual: individual.fitness, reverse=True)
        elite = population[0]
        return elite


def sample(environment):
    gym.logger.set_level(40)
    if isinstance(environment.action_space, gym.spaces.Discrete):
        action_space = list(range(environment.action_space.n))
        return random.choice(action_space)
    else:
        return [random.uniform(environment.action_space.low[0], environment.action_space.high[0])]
