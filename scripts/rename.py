""" Tool to rename files in folder """
#  Copyright (c) 2020, - All Rights Reserved
#  This file is part of the Evolutionary Planning on a Learned World Model thesis.
#  Unauthorized copying of this file, via any medium is strictly prohibited without the consensus of the authors.
#  Written by Thor V.A.N. Olesen <thorolesen@gmail.com> & Dennis T.T. Nguyen <dennisnguyen3000@yahoo.dk>.

import os
path = '/WorldModelPlanning/data_white/'
new_name = "white_random_policy_"
for filename in os.listdir(path):
    if "rename" in filename:
        continue
    os.rename(os.path.join(path, filename), os.path.join(path, new_name+filename))
