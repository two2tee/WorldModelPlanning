import time

import gym
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from gym.envs.box2d.car_dynamics import Car
from gym.envs.box2d.car_racing import CarRacing
import matplotlib
import copy

from planning.interfaces.individual import Individual
from utility.visualizer import Visualizer

matplotlib.use('Qt5Agg')  # Required for Python, Matplotlib 3 on Mac OSX

action = [0, 0, 0]
def on_key_press(event):
    """ Defines key pressed behavior """
    if event.key == 'up':
        action[1] = 0.3
    if event.key == 'down':
        action[2] = .8
    if event.key == 'left':
        action[0] = -1
    if event.key == 'right':
        action[0] = 1

def on_key_release(event):
    """ Defines key pressed behavior """
    if event.key == 'up':
        action[1] = 0
    if event.key == 'down':
        action[2] = 0
    if event.key == 'left' and action[0] == -1:
        action[0] = 0
    if event.key == 'right' and action[0] == 1:
        action[0] = 0


figure = plt.figure()
resized = (64, 64, 3)
monitor = plt.imshow(X=np.zeros(resized, dtype=np.uint8))
figure.canvas.mpl_connect('key_press_event', lambda event: on_key_press(event))
figure.canvas.mpl_connect('key_release_event', lambda event: on_key_release(event))






env = gym.make('CarRacing-v0')

start_track = 222
seed = 2 #9214
env = gym.make('CarRacing-v0')
env.seed(seed)
frame = env.reset()
env.env.car = Car(env.env.world, *env.env.track[start_track][1:4])


for _ in range(50):
    frame, _, _, _ = env.step(action)
    env.render()
#

while True:
    print(action)
    frame, _, _, _ = env.step(action)
    env.render()
    plt.pause(0.1)








