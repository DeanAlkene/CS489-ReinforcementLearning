import numpy as np
import matplotlib.pyplot as plt
import gym
import torch

class Enviroment:
    def __init__(self):
        self.env = gym.make('MountainCar-v0').unwrapped

    def reset(self):
        return self.env.reset()

    def render(self):
        self.env.render()

    def step(self, action):
        return self.env.step(action)
    
    def getActionSpace(self):
        return self.env.action_space

    def printEnvInfo(self):
        print("action space")
        print(env.action_space)
        print("observation space")
        print(env.observation_space)
        print("High")
        print(env.observation_space.high)
        print("Low")
        print(env.observation_space.low)

class Agent:
    def __init__(self):
        pass

    def getAction(self, ifEpsilonGreedy):
        pass

class ReplayBuffer:
    def __init__(self):
        pass

class DQN:
    def __init__(self):
        self.lr = 0.0001

env = gym.make('MountainCar-v0')
for i in range(10):
    observation = env.reset()
    t = 0
    done = False
    while not done:
        env.render()
        # print(observation)
        action = env.action_space.sample()
        print(action)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode %d ended after %d timesteps" % (i + 1, t + 1))
            break
        t += 1
env.close()
