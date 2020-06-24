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

class OrnsteinUhlenbeckActionNoise:
	def __init__(self, actionSize, mu = 0, theta = 0.15, sigma = 0.2):
		self.action_dim = actionSize
		self.mu = mu
		self.theta = theta
		self.sigma = sigma
		self.X = np.ones(self.action_dim) * self.mu

	def reset(self):
		self.X = np.ones(self.action_dim) * self.mu

	def sample(self):
		dx = self.theta * (self.mu - self.X)
		dx = dx + self.sigma * np.random.randn(len(self.X))
		self.X = self.X + dx
		return self.X

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0 #for recursive array

    def __len__(self):
        return len(self.buffer)

    def push(self, *args):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = Exp(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batchSize):
        return random.sample(self.buffer, batchSize)
    
    def fill(self, env, initLen, maxSteps):
        stateSize = env.observation_space.shape[0]
        if initLen > self.capacity:
            return
        while len(self.buffer) < initLen:
            state = env.reset()
            for t in count():
                action = env.action_space.sample()
                nextState, reward, done, _ = env.step(action)
                self.push(torch.tensor(state.reshape(1, stateSize), device=device, dtype=torch.float),
                          torch.tensor(action.reshape(1, -1), device=device, dtype=torch.float),
                          torch.tensor(np.array([reward]).reshape(1, -1), device=device, dtype=torch.float),
                          torch.tensor(nextState.reshape(1, stateSize), device=device, dtype=torch.float),
                          torch.tensor(np.array([not done]).reshape(1, -1), device=device, dtype=torch.long))
                state = nextState
                if done or t + 1 >= maxSteps:
                    break

class Actor(nn.Module):
    def __init__(self, stateSize, hiddenSize, actionSize, lim):
        super(Actor, self).__init__()
        self.lim = lim
        self.linear1 = nn.Linear(stateSize, hiddenSize)
        self.linear2 = nn.Linear(hiddenSize, hiddenSize // 2)
        self.linear3 = nn.Linear(hiddenSize // 2, actionSize)
    
    def forward(self, state):
        x = F.leaky_relu(self.linear1(state))
        x = F.leaky_relu(self.linear2(x))
        action = torch.tanh(self.linear3(x))
        action = action * self.lim
        return action

class Critic(nn.Module):
    def __init__(self, stateSize, hiddenSize, actionSize):
        super(Critic, self).__init__()
        self.state1 = nn.Linear(stateSize, hiddenSize)
        self.state2 = nn.Linear(hiddenSize, hiddenSize // 2)
        self.action1 = nn.Linear(actionSize, hiddenSize // 2)

        self.linear1 = nn.Linear(hiddenSize, hiddenSize // 2)
        self.linear2 = nn.Linear(hiddenSize // 2, 1)

    def forward(self, state, action):
        s1 = F.leaky_relu(self.state1(state))
        s2 = F.leaky_relu(self.state2(s1))
        a1 = F.leaky_relu(self.action1(action))

        x = torch.cat((s2, a1), dim=1)
        x = F.leaky_relu(self.linear1(x))
        x = self.linear2(x)
        return x

class DDPG:
    def __init__(self, env, buf, actorLR, criticLR, gamma, tau, batchSize, hiddenSize, maxSteps, maxEps, updateStride):
        self.env = env
        self.buffer = buf
        self.actorLR = actorLR
        self.criticLR = criticLR
        self.gamma = gamma
        self.tau = tau
        self.batchSize = batchSize
        self.hiddenSize = hiddenSize
        self.maxSteps = maxSteps
        self.maxEps = maxEps
        self.noise = OrnsteinUhlenbeckActionNoise(self.env.action_space.shape[0])
        self.updateStride = updateStride

        self.actor = Actor(self.env.observation_space.shape[0], self.hiddenSize, self.env.action_space.shape[0], self.env.action_space.high[0]).to(device)
        self.actorTarget = Actor(self.env.observation_space.shape[0], self.hiddenSize, self.env.action_space.shape[0], self.env.action_space.high[0]).to(device)
        self.critic = Critic(self.env.observation_space.shape[0], self.hiddenSize, self.env.action_space.shape[0]).to(device)
        self.criticTarget = Critic(self.env.observation_space.shape[0], self.hiddenSize, self.env.action_space.shape[0]).to(device)

        self.actorOpt = optim.Adam(self.actor.parameters(), self.actorLR)
        self.criticOpt = optim.Adam(self.critic.parameters(), self.criticLR)

        self._updateParams(self.actorTarget, self.actor)
        self._updateParams(self.criticTarget, self.critic)
    
    def _updateParams(self, t, s, isSoft=False):
        for t_param, param in zip(t.parameters(), s.parameters()):
            if isSoft:
                t_param.data.copy_(t_param.data * (1.0 - self.tau) + param.data * self.tau)
            else:
                t_param.data.copy_(param.data)
    
    def _getAction(self, state):
        action = self.actor.forward(torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float)).detach() + torch.tensor(self.noise.sample() * self.env.action_space.high[0], device=device, dtype=torch.float)
        return action.cpu().numpy()
    
    def learn(self):
        if len(self.buffer) < self.batchSize:
            print("Can't fetch enough exp!")
            return
        exps = self.buffer.sample(self.batchSize)
        batch = Exp(*zip(*exps))  # batch => Exp of batch
        stateBatch = torch.cat(batch.state) # batchSize * stateSpace.shape[0]
        actionBatch = torch.cat(batch.action) # batchSize * 1
        rewardBatch = torch.cat(batch.reward)  # batchSize * 1
        nextStateBatch = torch.cat(batch.nextState)  # batchSize * stateSpace.shape[0]
        doneBatch = torch.cat(batch.done)

        nextActionBatch = self.actor.forward(nextStateBatch).detach()
        targetQ = self.criticTarget.forward(nextStateBatch, nextActionBatch).detach().view(-1)
        y = rewardBatch.view(-1) + doneBatch.view(-1) * self.gamma * targetQ
        Q = self.critic.forward(stateBatch, actionBatch).view(-1)
        critic_loss = F.smooth_l1_loss(Q, y)

        self.criticOpt.zero_grad()
        critic_loss.backward()
        self.criticOpt.step()

        mu = self.actor.forward(stateBatch)
        actor_loss = -1.0 * torch.sum(self.critic.forward(stateBatch, mu))
        self.actorOpt.zero_grad()
        actor_loss.backward()
        self.actorOpt.step()

        self._updateParams(self.actorTarget, self.actor, isSoft=True)
        self._updateParams(self.criticTarget, self.critic, isSoft=True)

    def train(self):
        res = []
        N = 0
        for eps in range(self.maxEps):
            state = self.env.reset()
            ret = 0.0
            for t in count():
                #self.env.render()
                action = self._getAction(state)
                nextState, reward, done, _ = self.env.step(action)
                self.buffer.push(torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float),
                                 torch.tensor(action.reshape(1, -1), device=device, dtype=torch.float),
                                 torch.tensor(np.array([reward]).reshape(1, -1), device=device, dtype=torch.float),
                                 torch.tensor(nextState.reshape(1, -1), device=device, dtype=torch.float),
                                 torch.tensor(np.array([not done]).reshape(1, -1), device=device, dtype=torch.long))
                if N % self.updateStride and N > 0:
                    self.learn()
                ret += reward
                N += 1
                if done or t + 1 >= self.maxSteps:
                    print("Eps: %d\tRet: %f\tSteps: %d" % (eps + 1, ret, t + 1))
                    self.noise.reset()
                    break
                state = nextState
            res.append(ret)

        np.save('resDDPG', res)
        plt.plot(res)
        plt.ylabel('Return')
        plt.xlabel('Episodes')
        plt.savefig('resDDPG.png')
        torch.save(self.actor.state_dict(), 'netDDPG_A.pkl')
        torch.save(self.critic.state_dict(), 'netDDPG_C.pkl')
        self.env.close()

def runDDPG(env_name):
    env = gym.make(env_name)
    buf = ReplayBuffer(1000000)
    buf.fill(env, 1000, 200)
    exps = buf.sample(1000)
    ddpg = DDPG(env=env,
                buf=buf,
                actorLR=1e-4,
                criticLR=1e-3,
                gamma=0.99,
                tau=0.001,
                batchSize=64,
                hiddenSize=512,
                maxSteps=1000,
                maxEps=10000,
                updateStride=30)
    ddpg.train()


