""" RHEA/RMHC Test suite on simple environments """
#  Copyright (c) 2020, - All Rights Reserved
#  This file is part of the Evolutionary Planning on a Learned World Model thesis.
#  Unauthorized copying of this file, via any medium is strictly prohibited without the consensus of the authors.
#  Written by Thor V.A.N. Olesen <thorolesen@gmail.com> & Dennis T.T. Nguyen <dennisnguyen3000@yahoo.dk>.

import gym
import math
import time
import random
from planning.real.mcts import MCTS
from planning.real.rolling_horizon import RHEA
from planning.real.random_mutation_hill_climbing import RMHC

SEED = 0
is_seed = True
is_replay = False
MAX_TIME_STEPS = 200
games = ["FrozenLake-v0", "CartPole-v0", "Pendulum-v0"]

random_mutation_hill_climb_test_parameters = {
    "FrozenLake-v0": (25, 10, True,  False, 0.1) if is_seed else (25,   100, True,  False, 0.2),
    "CartPole-v0":   (1,   5, False, True,  0.1) if is_seed else (2,    50,  False, True),
    "Pendulum-v0":   (20, 30, True,  False, 0.1) if is_seed else (20,   30,  True,  False),
}

rolling_horizon_test_parameters = {
    "FrozenLake-v0": (4, 15, 6, True, False) if is_seed else (8, 20, 40, True,  False, 0.2),
    "CartPole-v0":   (4, 3,  10, True, True)  if is_seed else (8, 2,  10, True,  True),
    "Pendulum-v0":   (8, 20, 5, True, False) if is_seed else (8, 20, 10, True,  False),
}

monte_carlo_tree_search_parameters = {
    "FrozenLake-v0": (1, 10) if is_seed else (1, 50),
    "CartPole-v0":   (math.sqrt(2), 3) if is_seed else (1, 10),
    "Pendulum-v0":   (math.sqrt(2), 30) if is_seed else (math.sqrt(2), 30),
}

print(f" \nSTARTING PLANNING TESTS on {games}\n")
for game in games:

    agents = [RMHC(*random_mutation_hill_climb_test_parameters[game]),
              RHEA(*rolling_horizon_test_parameters[game]),
              MCTS(*monte_carlo_tree_search_parameters[game])]

    for agent in agents:
        print(f"TESTING {agent} on {game}")
        env = gym.make(game, is_slippery=False) if game == "FrozenLake-v0" else gym.make(game)
        if is_seed:
            env.seed(SEED)
            random.seed(SEED)
        env.reset()
        total_reward = 0
        actions = []
        start_time = time.time()
        for t in range(MAX_TIME_STEPS):
            action = agent.search(env)
            actions.append(action)

            _, reward, done, _ = env.step(action)
            total_reward += reward

            if done:
                if game == "FrozenLake-v0":
                    assert(total_reward == 1)
                elif game == "CartPole-v0":
                    print(total_reward)
                    assert(total_reward == 200)
                else:
                    print(total_reward)
                    expected_min_reward = -600 if isinstance(agent, MCTS) else -400
                    assert(total_reward >= expected_min_reward)
                print(f"Total Reward:{total_reward} | Reward: {reward} | Steps: {t} | Seconds: {round(time.time() - start_time, 2)} \n")
                break

        if is_replay:
            print(f"Replaying {agent} plan in {game} \n")
            if is_seed:
                random.seed(SEED)
                env.seed(SEED)
            env.reset()
            [(env.step(action), env.render()) for action in actions]
        env.close()
    print(f"SUCCESS: all agents passed {game} \n")

print(f"SUCCESS: all agents passed planning on {games} \n")
