""" RMHC for non-simulated environments """
#  Copyright (c) 2020, - All Rights Reserved
#  This file is part of the Evolutionary Planning on a Learned World Model thesis.
#  Unauthorized copying of this file, via any medium is strictly prohibited without the consensus of the authors.
#  Written by Thor V.A.N. Olesen <thorolesen@gmail.com> & Dennis T.T. Nguyen <dennisnguyen3000@yahoo.dk>.

import gym
import copy
import random
from planning.interfaces.individual import Individual
from planning.interfaces.abstract_hill_climb import AbstractRandomMutationHillClimbing


class RMHC(AbstractRandomMutationHillClimbing):
    def __init__(self, horizon, max_generations, is_shift_buffer, is_rollout, mutation_probability=None):
        super().__init__(horizon, max_generations, is_shift_buffer, is_rollout, mutation_probability)
        self.current_elite = None

    def search(self, environment):
        self.current_elite = self._initialize_individual(environment)
        best_fitness = self._evaluate_individual(self.current_elite, environment)

        for generation in range(self.max_generations):
            mutated_individual = self._mutate(environment, self.current_elite, generation)
            best_fitness = self._select_best_individual(best_fitness, mutated_individual, environment)

        return self.current_elite.action_sequence[0]

    def _initialize_individual(self, environment):
        if self.is_shift_buffer and self.current_elite is not None:
            individual = self._shift_buffer(environment, self.current_elite)
        else:
            individual = Individual(action_sequence=[sample(environment) for _ in range(self.horizon)])
        individual.fitness, individual.age = 0, 0  # reset across generations
        return individual

    def _select_best_individual(self, current_elite_fitness, mutated_individual, environment):
        mutated_individual_fitness = self._evaluate_individual(mutated_individual, environment)
        self.current_elite = mutated_individual if mutated_individual_fitness > current_elite_fitness else self.current_elite
        return self.current_elite.fitness

    def _shift_buffer(self, environment, individual):
        individual.action_sequence.pop(0)
        individual.action_sequence.append(sample(environment))
        return individual

    def _rollout(self, environment, is_done):
        total_reward = 0
        while not is_done:
            action = sample(environment)
            _, reward, is_done, _ = environment.step(action)
            total_reward += reward

        return total_reward

    def _mutate(self, environment, current_elite, generation):
        individual = copy.deepcopy(current_elite)

        mutation_indices = {random.randrange(self.horizon)}  # At least one gene mutation
        other_mutation_indices = set([i for i, _ in enumerate(individual.action_sequence)
                                      if random.uniform(0, 1) < self.mutation_probability])
        mutation_indices = mutation_indices.union(other_mutation_indices)
        mutations = [sample(environment) for _ in range(len(mutation_indices))]
        for i in mutation_indices:
            individual.action_sequence[i] = mutations.pop(0)

        individual.age, individual.fitness = generation + 1, 0
        return individual

    def _evaluate_individual(self, individual, environment):
        is_done = False
        total_reward = 0
        environment = copy.deepcopy(environment)  # Evaluate elite and mutation separately

        for action in individual.action_sequence:
            if not is_done:
                _, reward, is_done, _ = environment.step(action)
                total_reward += reward
            else:
                break

        if self.is_rollout and not is_done:
            total_reward += self._rollout(environment, is_done) if self.is_rollout else 0
        individual.fitness += total_reward

        return individual.fitness


def sample(environment):
    gym.logger.set_level(40)
    if isinstance(environment.action_space, gym.spaces.Discrete):
        action_space = list(range(environment.action_space.n))
        return random.choice(action_space)
    else:
        return [random.uniform(environment.action_space.low[0], environment.action_space.high[0])]
