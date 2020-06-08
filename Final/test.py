import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import math
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from itertools import count
import A3C
import DDPG

def testDDPG(env):
    net = DDPG.Actor(env.observation_space.shape[0], 512, env.action_space.shape[0], env.action_space.high[0])
    net.load_state_dict(torch.load('DDPG/Hopper2/netDDPG_A.pkl'))
    for i in range(100):
        state = env.reset()
        for t in count():
            env.render()
            action = net.forward(torch.tensor(state.reshape(1, -1), dtype=torch.float))
            nextState, reward, done, _ = env.step(action.detach().numpy())
            state = nextState
            if done:
                print("Episode %d ended in %d steps" % (i + 1, t + 1))
                break

env = gym.make('Hopper-v2')
testDDPG(env)