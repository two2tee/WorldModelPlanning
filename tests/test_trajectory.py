""" Tests for trajectory tracing """
#  Copyright (c) 2020, - All Rights Reserved
#  This file is part of the Evolutionary Planning on a Learned World Model thesis.
#  Unauthorized copying of this file, via any medium is strictly prohibited without the consensus of the authors.
#  Written by Thor V.A.N. Olesen <thorolesen@gmail.com> & Dennis T.T. Nguyen <dennisnguyen3000@yahoo.dk>.

import math
import gym
import numpy as np
from gym.envs.box2d.car_dynamics import Car
from utility.visualizer import Visualizer

SCALE = 6.0  # Track scale
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40 / SCALE
ROAD_COLOR = [0.4, 0.4, 0.4]


def recreate_tiles(env):
    # Red-white border on hard turns
    border = [False] * len(env.track)
    for i in range(len(env.track)):
        good = True
        oneside = 0
        for neg in range(BORDER_MIN_COUNT):
            beta1 = env.track[i - neg - 0][1]
            beta2 = env.track[i - neg - 1][1]
            good &= abs(beta1 - beta2) > TRACK_TURN_RATE * 0.2
            oneside += np.sign(beta1 - beta2)
        good &= abs(oneside) == BORDER_MIN_COUNT
        border[i] = good
    for i in range(len(env.track)):
        for neg in range(BORDER_MIN_COUNT):
            border[i - neg] |= border[i]

    # Create tiles
    for i in range(len(env.track)):
        alpha1, beta1, x1, y1 = env.track[i]
        alpha2, beta2, x2, y2 = env.track[i - 1]
        road1_l = (x1 - TRACK_WIDTH * math.cos(beta1), y1 - TRACK_WIDTH * math.sin(beta1))
        road1_r = (x1 + TRACK_WIDTH * math.cos(beta1), y1 + TRACK_WIDTH * math.sin(beta1))
        road2_l = (x2 - TRACK_WIDTH * math.cos(beta2), y2 - TRACK_WIDTH * math.sin(beta2))
        road2_r = (x2 + TRACK_WIDTH * math.cos(beta2), y2 + TRACK_WIDTH * math.sin(beta2))
        vertices = [road1_l, road1_r, road2_r, road2_l]
        env.fd_tile.shape.vertices = vertices
        t = env.world.CreateStaticBody(fixtures=env.fd_tile)
        t.userData = t
        c = 0.01 * (i % 3)
        t.color = [ROAD_COLOR[0] + c, ROAD_COLOR[1] + c, ROAD_COLOR[2] + c]
        t.road_visited = False
        t.road_friction = 1.0
        t.fixtures[0].sensor = True
        env.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
        env.road.append(t)
        if border[i]:
            side = np.sign(beta2 - beta1)
            b1_l = (x1 + side * TRACK_WIDTH * math.cos(beta1), y1 + side * TRACK_WIDTH * math.sin(beta1))
            b1_r = (x1 + side * (TRACK_WIDTH + BORDER) * math.cos(beta1),
                    y1 + side * (TRACK_WIDTH + BORDER) * math.sin(beta1))
            b2_l = (x2 + side * TRACK_WIDTH * math.cos(beta2), y2 + side * TRACK_WIDTH * math.sin(beta2))
            b2_r = (x2 + side * (TRACK_WIDTH + BORDER) * math.cos(beta2),
                    y2 + side * (TRACK_WIDTH + BORDER) * math.sin(beta2))
            env.road_poly.append(([b1_l, b1_r, b2_r, b2_l], (1, 1, 1) if i % 2 == 0 else (1, 0, 0)))


seed = 2
start_track = 205
env = gym.make('CarRacing-v0')
env.seed(seed)  # tiles
env.reset()
[env.step([0, 0, 0]) for _ in range(50)]  # env.ZOOM_FOLLOW = False not working
env.render()
env.env.car = Car(env.env.world, *env.env.track[start_track][1:4])
vis = Visualizer()

prior_action = [0, 0.2, 0]
STEPS_SO_FAR = 20
HORIZON = 75
action = [0, 1, 0]
plan = [action for _ in range(HORIZON)]
FPS = 50

# SIZE = 0.02
# ENGINE_POWER = 100000000*SIZE*SIZE
# WHEEL_MOMENT_OF_INERTIA = 4000*SIZE*SIZE
# FRICTION_LIMIT = 1000000*SIZE*SIZE     # friction ~= mass ~= size^2 (calculated implicitly using density)
# dt = 1.0 / FPS
# [(env.step(prior_action), env.render()) for _ in range(STEPS_SO_FAR)]
for _ in range(STEPS_SO_FAR):
    env.step(prior_action)
    env.render()

car = env.env.car  # inertia, angularVelocity, angle, linearVelocity, position
start_velocity = (car.hull.linearVelocity[0], car.hull.linearVelocity[1])
start_speed = np.sqrt(np.square(car.hull.linearVelocity[0]) + np.square(car.hull.linearVelocity[1]))
start_position, start_angle, start_angular_velocity, start_inertia = (car.hull.position[0], car.hull.position[1]), car.hull.angle, car.hull.angularVelocity, car.hull.inertia
start_time, start_reward, start_previous_reward, start_tile_visited_count = env.t, env.reward, env.prev_reward, env.tile_visited_count
print(f"\nInitial speed: {start_speed} | velocity: {start_velocity} | position: {start_position} | Angle, Angular, Inertia: {start_angle, start_angular_velocity, start_inertia}")

current_num_tiles_visited = start_tile_visited_count
total_tile_reward = 0
old_score = env.score_label.text

# START WHEEL INFORMATION: angularVelocity, linearVelocity, Omega, Phase, Position (and worldCenter)
def get_wheel_info(wheel):
    # OTHER: gravityScale, localCenter, mass, wheel_rad, worldCenter
    return {"angle": wheel.angle,
            "angular_damping": wheel.angularDamping,
            "angular_velocity": wheel.angularVelocity,
            "brake": wheel.brake,
            "gas": wheel.gas,
            "steer": wheel.steer,
            "inertia": wheel.inertia,
            "linear_damping": wheel.linearDamping,
            "linear_velocity": (wheel.linearVelocity[0], wheel.linearVelocity[1]),
            "omega": wheel.omega,
            "phase": wheel.phase,
            "position": (wheel.position[0], wheel.position[1])}


def set_wheel_info(wheel, wheel_info):
    wheel.angle = wheel_info['angle']
    wheel.angularDamping = wheel_info['angular_damping']
    wheel.angularVelocity = wheel_info['angular_velocity']
    wheel.brake = wheel_info['brake']
    wheel.gas = wheel_info['gas']
    wheel.steer = wheel_info['steer']
    wheel.inertia = wheel_info['inertia']
    wheel.linearDamping = wheel_info['linear_damping']
    wheel.linearVelocity = wheel_info['linear_velocity']
    wheel.omega = wheel_info['omega']
    wheel.phase = wheel_info['phase']
    wheel.position = wheel_info['position']


start_wheels = [get_wheel_info(car.wheels[wheel_num]) for wheel_num in range(4)]
[print(wheel_info) for wheel_info in start_wheels]

total_tiles_visited = 0

for action in plan:  # SIMULATION -> want to roll wheels back
    # Physics
    # displacement_x = car.hull.position[0] - start_position[0]
    # displacement_y = car.hull.position[1] - start_position[1]
    # distance = math.sqrt(displacement_x ** 2 + displacement_y ** 2)
    # print(displacement_x, displacement_y, distance)

    # Planning replace env.step(action)
    # env.step(action)
    car.steer(-action[0])
    car.gas(action[1])
    car.brake(action[2])
    car.step(1.0 / FPS)

    env.world.Step(1.0/FPS, 6*30, 2*30)
    env.t += 1.0 / FPS
    new_num_tiles = env.tile_visited_count - current_num_tiles_visited
    total_tiles_visited += new_num_tiles
    total_tile_reward += new_num_tiles * (1000.0 / len(env.track))

    env.reward += -0.1 + new_num_tiles * (1000.0/len(env.track))
    env.prev_reward = env.reward
    current_num_tiles_visited = env.tile_visited_count

    env.render()

    gas = action[1]
    next_velocity = (car.hull.linearVelocity[0], car.hull.linearVelocity[1])
    print(next_velocity)
    next_speed = np.sqrt(np.square(next_velocity[0]) + np.square(next_velocity[1]))

    # predicted_next_speed = dt * ENGINE_POWER * gas / WHEEL_MOMENT_OF_INERTIA / (abs(start_speed) + 5.0)  # v = v + dt * a
    # print(f"Action {action} | Target Speed: {next_speed} | Velocity: {car.hull.linearVelocity}")
    # print(f"Predicted Speed: {predicted_next_speed} | Diff: {abs(round(next_speed - predicted_next_speed, 2))} \n")
    next_position, next_angle, next_angular_velocity, next_inertia = (car.hull.position[0], car.hull.position[1]), car.hull.angle, car.hull.angularVelocity, car.hull.inertia
print(f"\nNext speed: {next_speed} | Next Velocity: {next_velocity} | Next Position: {next_position} | Angle, Angular, Inertia: {next_angle, next_angular_velocity, next_inertia}")

# Reset env (reward, prev_reward, tile_visited_count, t) and car back to original checkpoint
env.t = start_time
env.reward = start_reward
env.prev_reward = start_previous_reward
env.tile_visited_count = start_tile_visited_count
recreate_tiles(env)
# print(f"Reset rewards: {env.reward, env.prev_reward}")

# APPROACH: HOW TO TRANSFER WHEELS TO NEW CAR

next_wheels = [get_wheel_info(car.wheels[wheel_num]) for wheel_num in range(4)]
[print(wheel_info) for wheel_info in next_wheels]

car.destroy()
env.env.car = Car(world=env.env.world, init_angle=start_angle, init_x=start_position[0], init_y=start_position[1])
# car.__init__(env.env.world, start_angle, start_position[0], start_position[1])

print("RESET WHEELS \n")
[set_wheel_info(env.env.car.wheels[i], start_wheels[i]) for i in range(4)]
reset_wheels = [get_wheel_info(env.env.car.wheels[wheel_num]) for wheel_num in range(4)]
[print(wheel_info) for wheel_info in reset_wheels]
env.env.car.hull.linearVelocity[0], env.env.car.hull.linearVelocity[1] = start_velocity[0], start_velocity[1]
env.env.car.hull.angularVelocity = start_angular_velocity

# Verify env and car reset
after_velocity, after_position, after_angle, after_angular_velocity, after_inertia = env.env.car.hull.linearVelocity, env.env.car.hull.position, env.env.car.hull.angle, env.env.car.hull.angularVelocity, env.env.car.hull.inertia
after_speed = np.sqrt(np.square(after_velocity[0]) + np.square(after_velocity[1]))
print(f"\nAfter Reset speed: {after_speed} | Velocity: {after_velocity} | Position: {after_position} | Angle, Angular, Inertia: {after_angle, after_angular_velocity, after_inertia}")
assert after_speed == start_speed  # Ensure same speed
assert after_velocity == start_velocity  # Ensure same velocity
assert after_position == start_position  # Ensure same car position
assert env.tile_visited_count == start_tile_visited_count  # Ensure not done
assert env.reward == start_reward
assert env.prev_reward == start_previous_reward
assert reset_wheels[0] == start_wheels[0]
assert reset_wheels[1] == start_wheels[1]
assert reset_wheels[2] == start_wheels[2]
assert reset_wheels[3] == start_wheels[3]

# env.render()
env.score_label.text = "%04i" % start_reward  # Total reward on screen does not match real total reward above (nice to have: reset tile is visited to false)
env.score_label.draw()
# env.step([0, 0, 0])
env.render()

# Found elite -> Execute final plan
for action in plan:
    obs, reward, done, _ = env.step(action)
    env.render()

env.reward = env.reward + total_tiles_visited * 1000.0/len(env.track)
print(f"\nReal total reward: {env.reward}")

# TODO 1 : Extract next_velocity
# TODO 2:  Merge with Dennis angle velocity
# TODO 3:  Plot from origin=initial car position where origin is mutated to be new position after velocity/angle applied










