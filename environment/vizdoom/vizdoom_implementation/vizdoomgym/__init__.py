import gym
from gym.envs.registration import register
from environment.vizdoom.vizdoom_implementation.vizdoomgym.vizdoomenv import VizdoomEnv
from environment.vizdoom.vizdoom_implementation.vizdoomgym.vizdoom_env_definitions import (
    VizdoomBasic,
    VizdoomCorridor,
    VizdoomDefendCenter,
    VizdoomDefendLine,
    VizdoomHealthGathering,
    VizdoomMyWayHome,
    VizdoomPredictPosition,
    VizdoomTakeCover,
    VizdoomDeathmatch,
    VizdoomHealthGatheringSupreme,
)

# try registrate basic - if already exists in gym, clear all and registrate all again to avoid errors
try:
    register(
        id='VizdoomBasic-v0',
        entry_point='environment.vizdoom.vizdoom_implementation.vizdoomgym:VizdoomBasic'
    )
except:
    env_dict = gym.envs.registration.registry.env_specs.copy()
    for env in env_dict:
        if 'Vizdoom' in env:
            print("Remove {} from registry".format(env))
            del gym.envs.registration.registry.env_specs[env]

register(
    id='VizdoomCorridor-v0',
    entry_point='environment.vizdoom.vizdoom_implementation.vizdoomgym:VizdoomCorridor'
)

register(
    id='VizdoomDefendCenter-v0',
    entry_point='environment.vizdoom.vizdoom_implementation.vizdoomgym:VizdoomDefendCenter'
)

register(
    id='VizdoomDefendLine-v0',
    entry_point='environment.vizdoom.vizdoom_implementation.vizdoomgym:VizdoomDefendLine'
)

register(
    id='VizdoomHealthGathering-v0',
    entry_point='environment.vizdoom.vizdoom_implementation.vizdoomgym:VizdoomHealthGathering'
)

register(
    id='VizdoomMyWayHome-v0',
    entry_point='environment.vizdoom.vizdoom_implementation.vizdoomgym:VizdoomMyWayHome'
)

register(
    id='VizdoomPredictPosition-v0',
    entry_point='environment.vizdoom.vizdoom_implementation.vizdoomgym:VizdoomPredictPosition'
)

register(
    id='VizdoomTakeCover-v0',
    entry_point='environment.vizdoom.vizdoom_implementation.vizdoomgym:VizdoomTakeCover'
)

register(
    id='VizdoomDeathmatch-v0',
    entry_point='environment.vizdoom.vizdoom_implementation.vizdoomgym:VizdoomDeathmatch'
)

register(
    id='VizdoomHealthGatheringSupreme-v0',
    entry_point='environment.vizdoom.vizdoom_implementation.vizdoomgym:VizdoomHealthGatheringSupreme'
)
