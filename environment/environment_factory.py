from environment.carracing.car_racing_environment import CarRacingEnvironment
# from environment.vizdoom.vizdoom_environment import VizdoomEnvironment


def get_environment(config):
    if config['game'] == 'CarRacing-v0':
        return CarRacingEnvironment(config)
    if config['game'] == 'vizdoom-v0':
        # return VizdoomEnvironment(config)
        pass
    raise Exception(f'No implementation of action sampler was found for game: {config["game"]}')
