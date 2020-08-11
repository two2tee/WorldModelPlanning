from utility.carracing.car_racing_rollout_generator import RolloutGenerator as CarRacingRolloutGenerator


def get_rollout_generator(config):
    if config['game'] == 'CarRacing-v0':
        return CarRacingRolloutGenerator(config, data_output_dir=config['data_generator']['car_racing']['data_output_dir'])
    if config['game'] == 'viz-doom':
        return NotImplemented()
    raise Exception(f'No implementation of rollout generator was found for game: {config["game"]}')
