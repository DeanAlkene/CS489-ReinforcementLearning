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
import DDPG_noise
import DQN
import time

def testDDPG(env_name):
    env = gym.make(env_name)
    net = DDPG.Actor(env.observation_space.shape[0], 512, env.action_space.shape[0], env.action_space.high[0])
    if env_name == 'HalfCheetah-v2':
        net.load_state_dict(torch.load('model/DDPG/HalfCheetah/net.pkl'))
    elif env_name == 'Hopper-v2':
        net.load_state_dict(torch.load('model/DDPG/Hopper/net.pkl'))
    else:
        return
    avg = 0.0
    res = []
    for i in range(100):
        state = env.reset()
        ret = 0.0
        for t in count():
            # env.render()
            action = net.forward(torch.tensor(state.reshape(1, -1), dtype=torch.float))
            nextState, reward, done, _ = env.step(action.detach().numpy())
            ret += reward
            state = nextState
            if env_name == 'HalfCheetah-v2':
                if t + 1 >= 1000:
                    done = True
            if done:
                print("Episode %d ended in %d steps" % (i + 1, t + 1))
                res.append(ret)
                break
    avg = np.average(res)
    return res, avg

def testDDPG_noise(env_name):
    env = gym.make(env_name)
    net = DDPG_noise.Actor(env.observation_space.shape[0], 512, env.action_space.shape[0], env.action_space.high[0])
    if env_name == 'HalfCheetah-v2':
        net.load_state_dict(torch.load('model/DDPG/HalfCheetah_noise/net.pkl'))
    elif env_name == 'Hopper-v2':
        net.load_state_dict(torch.load('model/DDPG/Hopper_noise/net.pkl'))
    else:
        return
    avg = 0.0
    res = []
    for i in range(100):
        state = env.reset()
        ret = 0.0
        for t in count():
            # env.render()
            action = net.forward(torch.tensor(state.reshape(1, -1), dtype=torch.float))
            nextState, reward, done, _ = env.step(action.detach().numpy())
            ret += reward
            state = nextState
            if env_name == 'HalfCheetah-v2':
                if t + 1 >= 1000:
                    done = True
            if done:
                print("Episode %d ended in %d steps" % (i + 1, t + 1))
                res.append(ret)
                break
    avg = np.average(res)
    return res, avg

def testA3C(env_name):
    env = gym.make(env_name)
    net = A3C.ACNet(env.observation_space.shape[0], 512, env.action_space.shape[0])
    if env_name == 'HalfCheetah-v2':
        net.load_state_dict(torch.load('model/A3C/HalfCheetah/net.pkl'))
    elif env_name == 'Hopper-v2':
        net.load_state_dict(torch.load('model/A3C/Hopper/net.pkl'))
    else:
        return
    avg = 0.0
    res = []
    for i in range(100):
        state = env.reset()
        ret = 0.0
        for t in count():
            # env.render()
            action = net.selectAction(torch.from_numpy(state.reshape(1, -1).astype(np.float32)))
            nextState, reward, done, _ = env.step(action)
            ret += reward
            state = nextState
            if env_name == 'HalfCheetah-v2':
                if t + 1 >= 1000:
                    done = True
            if done:
                print("Episode %d ended in %d steps" % (i + 1, t + 1))
                res.append(ret)
                break
    avg = np.average(res)
    return res, avg

def testDQN(env_name):
    env = DQN.getEnv(env_name)
    net = DQN.DuelingDQN(84, 84, 4, 512, env.action_space.n)
    if env_name == 'BreakoutNoFrameskip-v4':
        net.load_state_dict(torch.load('model/DQN/Breakout/net.pkl'))
    elif env_name == 'PongNoFrameskip-v4':
        net.load_state_dict(torch.load('model/DQN/Pong/net.pkl'))
    else:
        return
    avg = 0.0
    res = []
    for i in range(100):
        state = np.array(env.reset())
        ret = 0.0
        for t in count():
            env.render()
            action = net(DQN.resize(state).unsqueeze(0).float()).max(1)[1].view(-1)
            state, reward, done, _ = env.step(action.item())
            ret += reward
            state = np.array(state)
            if done:
                print("%d, %f" % (t + 1, ret))
                res.append(ret)
                break
    
    avg = np.average(res)
    return res, avg

res, avg = testDDPG('HalfCheetah-v2')
plt.figure(figsize=(10, 8))
plt.plot(res)
plt.plot([0, 99], [avg, avg])
plt.xlabel('Eps')
plt.ylabel('Score')
plt.title('HalfCheetah-v2 Test Score')
plt.savefig('TestDDPGHC')

res, avg = testDDPG_noise('HalfCheetah-v2')
plt.figure(figsize=(10, 8))
plt.plot(res)
plt.plot([0, 99], [avg, avg])
plt.xlabel('Eps')
plt.ylabel('Score')
plt.title('HalfCheetah-v2 Test Score')
plt.savefig('TestDDPGHCN')

res, avg = testA3C('HalfCheetah-v2')
plt.figure(figsize=(10, 8))
plt.plot(res)
plt.plot([0, 99], [avg, avg])
plt.xlabel('Eps')
plt.ylabel('Score')
plt.title('HalfCheetah-v2 Test Score')
plt.savefig('TestA3CHC')
