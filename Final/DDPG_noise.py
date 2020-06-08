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

class AdaptiveParamNoiseSpec:
    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.2, adaptation_coefficient=1.01):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adaptation_coefficient = adaptation_coefficient
        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:
            self.current_stddev /= self.adaptation_coefficient
        else:
            self.current_stddev *= self.adaptation_coefficient
    
    def get_stats(self):
        stats = {
            'param_noise_stddev': self.current_stddev,
        }
        return stats

    def __repr__(self):
        fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adaptation_coefficient={})'
        return fmt.format(self.initial_stddev, self.desired_action_stddev, self.adaptation_coefficient)

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
        self.param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.05, desired_action_stddev=0.2, adaptation_coefficient=1.01)
        self.updateStride = updateStride

        self.actor = Actor(self.env.observation_space.shape[0], self.hiddenSize, self.env.action_space.shape[0], self.env.action_space.high[0]).to(device)
        self.actor_perturbed = Actor(self.env.observation_space.shape[0], self.hiddenSize, self.env.action_space.shape[0], self.env.action_space.high[0]).to(device)
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
    
    def _getAction(self, state, isTensor=False, isPerturbed=True):
        if isTensor:
            if isPerturbed:
                action = self.actor_perturbed.forward(state).detach() + torch.tensor(self.noise.sample() * self.env.action_space.high[0], device=device, dtype=torch.float)
            else:
                action = self.actor.forward(state).detach()
        else:
            if isPerturbed:
                action = self.actor_perturbed.forward(torch.tensor(state.reshape(-1, self.env.observation_space.shape[0]), device=device, dtype=torch.float)).detach() + torch.tensor(self.noise.sample() * self.env.action_space.high[0], device=device, dtype=torch.float)
            else:
                action = self.actor.forward(torch.tensor(state.reshape(-1, self.env.observation_space.shape[0]), device=device, dtype=torch.float)).detach()
        return action.cpu().numpy()

    def ddpg_distance_metric(self, actions1, actions2):
        diff = actions1 - actions2
        mean_diff = np.mean(np.square(diff), axis=0)
        dist = math.sqrt(np.mean(mean_diff))
        return dist

    def perturb_actor_parameters(self, param_noise):
        self._updateParams(self.actor_perturbed, self.actor)
        params = self.actor_perturbed.state_dict()
        for name in params:
            if 'ln' in name: 
                pass 
            param = params[name]
            random = torch.randn(param.shape).to(device)
            param += random * param_noise.current_stddev
    
    def adapt(self, noise_counter):
        if self.buffer.position - noise_counter > 0:
            noise_data = self.buffer.buffer[self.buffer.position - noise_counter : self.buffer.position]
        else:
            noise_data = self.buffer.buffer[self.buffer.position - noise_counter + self.buffer.capacity : self.buffer.capacity] + self.buffer.buffer[0: self.buffer.position]

        noise_data = np.array(noise_data)
        validBuf = Exp(*zip(*noise_data))
        noise_s = torch.cat(validBuf.state)
        noise_a = torch.cat(validBuf.action)
        perturbed_actions = noise_a.cpu().numpy()
        unperturbed_actions = self._getAction(noise_s, isTensor=True, isPerturbed=False)
        dist = self.ddpg_distance_metric(perturbed_actions, unperturbed_actions)
        self.param_noise.adapt(dist)

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
            noise_counter = 0
            self.perturb_actor_parameters(self.param_noise)
            for t in count():
                #self.env.render()
                self.noise.reset()
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
                noise_counter += 1
                if done or t + 1 >= self.maxSteps:
                    print("Eps: %d\tRet: %f\tSteps: %d" % (eps + 1, ret, t + 1))
                    break
                state = nextState
            self.adapt(noise_counter)
            res.append(ret)

        np.save('resDDPG', res)
        plt.plot(res)
        plt.ylabel('Return')
        plt.xlabel('Episodes')
        plt.savefig('resDDPG.png')
        torch.save(self.actor.state_dict(), 'netDDPG_A.pkl')
        torch.save(self.critic.state_dict(), 'netDDPG_C.pkl')
        self.env.close()

def main():
    env = gym.make('HalfCheetah-v2')
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

if __name__ == '__main__':
    main()