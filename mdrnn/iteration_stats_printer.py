#  Copyright (c) 2020, - All Rights Reserved
#  This file is part of the Evolutionary Planning on a Learned World Model thesis.
#  Unauthorized copying of this file, via any medium is strictly prohibited without the consensus of the authors.
#  Written by Thor V.A.N. Olesen <thorolesen@gmail.com> & Dennis T.T. Nguyen <dennisnguyen3000@yahoo.dk>.

import pickle
from os.path import exists
from iteration_stats.iteration_result import IterationResult


filename = 'iteration_stats/iterative_stats_World_Model_B.pickle'
if exists(filename):
    with open(f'{filename}', 'rb') as file:
        file_content = pickle.load(file)
        iteration_stats = [IterationResult.to_obj(encoded_result) for encoded_result in file_content['iteration_results']]
        [print(iteration_stat) for iteration_stat in iteration_stats]

else:
    print(f'File not found: {filename}')
