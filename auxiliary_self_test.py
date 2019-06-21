#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym
import numpy as np
# env = gym.make('Pendulum-v0')
env = gym.make('Acrobot-v1')

# env = gym.make('Alien-ram-v0')
env = env.unwrapped


# print(int("".join(list(filter(str.isdigit,str(env.action_space))))))
print(env.action_space.n)
# print(env.action_space.high)
# print(env.action_space.low)
print(env.observation_space.shape[0])
print(env.observation_space.high)
print(env.observation_space.low)
env.reset()
env.step(3)
# env.step(np.array([2]).astype(int))
# print((env.observation_space.high - env.observation_space.low)/2)