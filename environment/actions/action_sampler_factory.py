from environment.carracing.car_racing_action_sampler import CarRacingActionSampler
from environment.vizdoom.vizdoom_action_sampler import VizdoomActionSampler


def get_action_sampler(config):
    if config['game'] == 'CarRacing-v0':
        return CarRacingActionSampler(config)
    if config['game'] == 'viz-doom':
        return VizdoomActionSampler(config)
    raise Exception(f'No implementation of action sampler was found for game: {config["game"]}')