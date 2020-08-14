import gym
from environment.vizdoom.vizdoom_implementation import vizdoomgym
env = gym.make('VizdoomTakeCover-v0')

while True:
    state, reward, done, info = env.step(env.action_space.sample())
    env.render()
    print(reward, done, info)
    if done:
        env.reset()