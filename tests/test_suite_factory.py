#  Copyright (c) 2020, - All Rights Reserved
#  This file is part of the Evolutionary Planning on a Learned World Model thesis.
#  Unauthorized copying of this file, via any medium is strictly prohibited without the consensus of the authors.
#  Written by Thor V.A.N. Olesen <thorolesen@gmail.com> & Dennis T.T. Nguyen <dennisnguyen3000@yahoo.dk>.

from tests.car_racing.car_racing_model_tester import ModelTester as CarRacingModelTester
from tests.car_racing.car_racing_planning_tester import PlanningTester as CarRacingPlanningTester
from tests.viz_doom.vizdoom_planning_tester import VizDoomPlanningTester

def get_planning_tester(config, vae, mdrnn, preprocessor, environment, agent):
    if config['game'] == 'CarRacing-v0':
        return CarRacingPlanningTester(config, vae, mdrnn, preprocessor, environment, agent)
    if config['game'] == 'vizdoom-v0':
        return VizDoomPlanningTester(config, vae, mdrnn, preprocessor, environment, agent)
    raise Exception(f'No implementation of planning tester was found for game: {config["game"]}')


def get_model_tester(config, vae, mdrnn, preprocessor, environment):
    if config['game'] == 'CarRacing-v0':
        return CarRacingModelTester(config, vae, mdrnn, preprocessor, environment)
    if config['game'] == 'vizdoom-v0':
        return NotImplemented()
    raise Exception(f'No implementation of model tester was found for game: {config["game"]}')
