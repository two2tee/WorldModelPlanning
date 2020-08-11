""" Helper functions to estimate true car velocities for trajectory visualizations """
#  Copyright (c) 2020, - All Rights Reserved
#  This file is part of the Evolutionary Planning on a Learned World Model thesis.
#  Unauthorized copying of this file, via any medium is strictly prohibited without the consensus of the authors.
#  Written by Thor V.A.N. Olesen <thorolesen@gmail.com> & Dennis T.T. Nguyen <dennisnguyen3000@yahoo.dk>.

import math
import numpy as np
from gym.envs.box2d.car_dynamics import Car

FPS = 50
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


def get_wheel_info(wheel):
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


def get_trajectory(env, plan, is_best_elite, is_render=False):
    car = env.env.car
    start_velocity = (car.hull.linearVelocity[0], car.hull.linearVelocity[1])
    start_position, start_angle, start_angular_velocity, start_inertia = (car.hull.position[0], car.hull.position[1]), \
                                                                         car.hull.angle, car.hull.angularVelocity, car.hull.inertia
    start_wheels = [get_wheel_info(car.wheels[wheel_num]) for wheel_num in range(4)]

    velocities = []
    for i, action in enumerate(plan):
        action = get_action(action)
        car.steer(-action[0])
        car.gas(action[1])
        car.brake(action[2])
        car.step(1.0 / FPS)
        env.world.Step(1.0/FPS, 6*30, 2*30)

        if is_render:
            env.render()

        next_velocity = (abs(car.hull.linearVelocity[0] - start_velocity[0]), abs(car.hull.linearVelocity[1] - start_velocity[1]))
        velocities.append(next_velocity)

        if i == 0 and is_best_elite:  # Capture info to reset environment
            start_angle = car.hull.angle
            start_position = (car.hull.position[0], car.hull.position[1])
            start_wheels = [get_wheel_info(car.wheels[wheel_num]) for wheel_num in range(4)]
            start_velocity = (car.hull.linearVelocity[0], car.hull.linearVelocity[1])
            start_angular_velocity = car.hull.angularVelocity

    # Reset: Rollback environment and car to previous checkpoint before planning
    car.destroy()
    env.env.car = Car(world=env.env.world, init_angle=start_angle, init_x=start_position[0], init_y=start_position[1])
    [set_wheel_info(env.env.car.wheels[i], start_wheels[i]) for i in range(4)]
    env.env.car.hull.linearVelocity[0], env.env.car.hull.linearVelocity[1] = start_velocity[0], start_velocity[1]
    env.env.car.hull.angularVelocity = start_angular_velocity

    return velocities

def get_action(action):
    return action[0] if type(action) == tuple and len(action) == 2 else action