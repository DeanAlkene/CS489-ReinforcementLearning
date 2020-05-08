import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import random
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Exp = namedtuple('Exp', ('state', 'action', 'nextState', 'reward'))

class Enviroment:
    def __init__(self):
        self.env = gym.make('MountainCar-v0').unwrapped

    def reset(self):
        return self.env.reset()

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

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
    def __init__(self, actionSpace, observationSpace, gamma, epsilon, batchSize, lr, hiddenSize, updateStride):
        self.actionSpace = actionSpace
        self.actionNum = actionSpace.n
        self.actionSpace = [i for i in range(self.stateNum)]
        self.stateSize = observationSpace.shape[0]
        self.gamma = gamma
        self.epsilon = epsilon
        self.batchSize = batchSize
        self.lr = lr
        self.hiddenSize = hiddenSize
        self.updateStride = updateStride
        self.net = DQN(self.stateSize, self.hiddenSize, 1)
        self.targetNet = DQN(self.stateSize, self.hiddenSize, 1)
        self.optimizer = optim.Adam(param=self.Q.parameters, lr=self.lr)

    def getAction(self, state, ifEpsilonGreedy=True):
        if ifEpsilonGreedy:
            if random.random() > self.epsilon:
                with torch.no_grad():
                    return self.net(state).max(1)[1].view(1, 1)
            else:
                return torch.tensor([[random.randrange(self.actionNum)]], dtype=torch.long)
        else:
            with torch.no_grad():
                return self.net(state).max(1)[1].view(1, 1)
    
    def learn(self):
        pass

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0 #for recursive queue

    def __len__(self):
        return len(self.buffer)

    def push(self, *args):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = Exp(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batchSize):
        return random.sample(self.buffer, batchSize)
    
    def fill(self, env, initLen):
        pass


class DQN(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(inputSize, hiddenSize)
        self.fc2 = nn.Linear(hiddenSize, hiddenSize)
        self.fc3 = nn.Linear(hiddenSize, outputSize)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        # x = F.leaky_relu(self.fc1(x))
        # x = F.leaky_relu(self.fc2(x))
        # x = F.leaky_relu(self.fc3(x))
        return x

def main():
    pass

if __name__ == "__main__":
    main()

# env = gym.make('MountainCar-v0')
# for i in range(10):
#     observation = env.reset()
#     t = 0
#     done = False
#     while not done:
#         env.render()
#         # print(observation)
#         action = env.action_space.sample()
#         print(action)
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode %d ended after %d timesteps" % (i + 1, t + 1))
#             break
#         t += 1
# env.close()
