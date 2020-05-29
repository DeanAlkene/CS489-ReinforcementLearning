import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import namedtuple
from itertools import count
import random
import math
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Exp = namedtuple('Exp', ('state', 'action', 'reward', 'nextState', 'done'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Enviroment:
    def __init__(self):
        self.env = gym.make('Pendulum-v0').unwrapped

    def reset(self):
        return self.env.reset()

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        return state, reward, done, info

    def getActionSpace(self):
        return self.env.action_space

    def getObservationSpace(self):
        return self.env.observation_space

    def printEnvInfo(self):
        print("action space")
        print(env.action_space)
        print("observation space")
        print(env.observation_space)
        print("High")
        print(env.observation_space.high)
        print("Low")
        print(env.observation_space.low)

env = Enviroment()
state = env.reset()
for i in range(10):
    env.render()
    action = random.random() * 4 - 2
    nextState, reward, done, _ = env.step([action])
    print(state, action, reward, nextState)
    state = nextState

env.close()
