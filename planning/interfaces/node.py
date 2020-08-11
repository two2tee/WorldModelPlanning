""" Node representation for MCTS """
#  Copyright (c) 2020, - All Rights Reserved
#  This file is part of the Evolutionary Planning on a Learned World Model thesis.
#  Unauthorized copying of this file, via any medium is strictly prohibited without the consensus of the authors.
#  Written by Thor V.A.N. Olesen <thorolesen@gmail.com> & Dennis T.T. Nguyen <dennisnguyen3000@yahoo.dk>.

class Node:
    def __init__(self, parent=None, action=None, actions=None):
        self.action = action
        self.parent = parent
        self.total_reward = 0
        self.visit_count = 0  # N(s,a)
        self.children = []
        self.actions = actions

    def is_fully_expanded(self):
        return len(self.actions) <= 0
