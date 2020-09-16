""" Representation of individual for RHEA/RMHC """
#  Copyright (c) 2020, - All Rights Reserved
#  This file is part of the Evolutionary Planning on a Learned World Model thesis.
#  Unauthorized copying of this file, via any medium is strictly prohibited without the consensus of the authors.
#  Written by Thor V.A.N. Olesen <thorolesen@gmail.com> & Dennis T.T. Nguyen <dennisnguyen3000@yahoo.dk>.

class Individual:
    def __init__(self, action_sequence, age=0, fitness=0):
        self.action_sequence = action_sequence  # chromosome
        self.fitness = fitness
        self.age = age
        self.standard_deviation = 0

    def __str__(self):
        return f'Age: {self.age} | Fitness: {self.fitness} | Sequence: {self.action_sequence}'

    def get_full_sequence(self):
        return self.action_sequence


