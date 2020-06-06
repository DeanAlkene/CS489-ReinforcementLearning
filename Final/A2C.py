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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# try:
#     mp.set_start_method('spawn')
# except RuntimeError:
#     pass

class ACNet(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(ACNet, self).__init__()
        self.actor1 = nn.Linear(inputSize, hiddenSize)
        #self.actor2 = nn.Linear(hiddenSize // 2, hiddenSize // 2)
        self.mu = nn.Linear(hiddenSize, outputSize)
        self.sigma = nn.Linear(hiddenSize, outputSize)

        self.critic1 = nn.Linear(inputSize, hiddenSize // 2)
        #self.critic2 = nn.Linear(hiddenSize // 2, hiddenSize // 4)
        self.value = nn.Linear(hiddenSize // 2, 1)

        # for l in [self.linear1, self.actor1, self.actor2, self.mu, self.sigma, self.critic1, self.critic2, self.value]:
        for l in [self.actor1, self.mu, self.sigma, self.critic1, self.value]:
            self._initLayer(l)

        self.distribution = torch.distributions.Normal

    def _initLayer(self, layer):
        nn.init.normal_(layer.weight, mean=0.0, std=0.1)
        nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        # x = F.leaky_relu(self.linear1(inputs))
        actor1 = F.leaky_relu(self.actor1(x))
        # actor2 = F.leaky_relu(self.actor2(actor1))
        mu = F.leaky_relu(self.mu(actor1))
        sigma = F.softplus(self.sigma(actor1)) + 0.00001
        critic1 = F.leaky_relu(self.critic1(x))
        # critic2 = F.leaky_relu(self.critic2(critic1))
        value = self.value(critic1)
        return mu, sigma, value
    
    def loss(self, state, action, R):
        self.train()
        mu, sigma, value = self.forward(state)
        error = R - value
        critic_loss = error.pow(2)
        dist = self.distribution(mu, sigma)
        log_prob = dist.log_prob(action)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(dist.scale)
        actor_loss = -(log_prob * error.detach() + 0.0001 * entropy)
        return (critic_loss + actor_loss).mean()

    def selectAction(self, state):
        self.training = False
        with torch.no_grad():
            mu, sigma, _ = self.forward(state)
            dist = self.distribution(mu.detach(), sigma.detach())
            return dist.sample().cpu().numpy()

class A2C():
    def __init__(self, lr, maxEps, maxSteps, updateStride, gamma, hiddenSize):
        self.env = gym.make('Pendulum-v0').unwrapped
        self.Net = ACNet(self.env.observation_space.shape[0], hiddenSize, self.env.action_space.shape[0]).to(device)
        self.opt = optim.Adam(self.Net.parameters(), lr=lr, betas=(0.9, 0.99))
        self.params = {'MAX_EPISODE': maxEps, 'MAX_STEP': maxSteps, 'UPDATE_STRIDE': updateStride, 'gamma': gamma, 'hiddenSize': hiddenSize}

    def train(self):
        steps = 1
        totEps = 0
        res = []
        while totEps < self.params['MAX_EPISODE']:
            stateBuf, actionBuf, rewardBuf = [], [], []
            state = self.env.reset()
            ret = 0.0
            #rewardDecay = 1.0

            for t in range(self.params['MAX_STEP']):
                self.env.render()
                action = self.Net.selectAction(torch.from_numpy(state.reshape(1, -1).astype(np.float32)).to(device))
                nextState, reward, done, _ = self.env.step(action)
                if t == self.params['MAX_STEP'] - 1:
                    done = True
                ret += reward
                #ret += rewardDecay * reward
                #rewardDecay *= self.params['gamma']
                stateBuf.append(state.reshape(-1))
                actionBuf.append(action)
                rewardBuf.append(reward)
                if done:
                    loss = self.learn(nextState, done, stateBuf, actionBuf, rewardBuf)
                    stateBuf, actionBuf, rewardBuf = [], [], []

                    totEps += 1
                    res.append(ret)
                    print("Eps: %d\tTotRet: %f\tSteps: %d\tLoss: %f" % (totEps, ret, t + 1, loss))
                    if t + 1 <= 500:
                        self.params['UPDATE_STRIDE'] = 20
                    elif t + 1 > 500:
                        self.params['UPDATE_STRIDE'] = 50
                    elif t + 1 > 1000:
                        self.params['UPDATE_STRIDE'] = 100
                    break
                
                state = nextState
                steps += 1

        np.save('res', res)
        plt.plot(res)
        plt.ylabel('Average Return')
        plt.xlabel('Episodes')
        plt.savefig('res.png')
        torch.save(self.Net.state_dict(), 'net.pkl')
        self.env.close()

    def learn(self, nextState, done, stateBuf, actionBuf, rewardBuf):
        if done:
            R = 0
        else:
            R = self.Net.forward(torch.from_numpy(nextState.reshape(1, -1).astype(np.float32)).to(device))[-1][0].item()
        
        RBuf = []
        for r in rewardBuf[::-1]:
            R = r + self.params['gamma'] * R
            RBuf.append(R)
        RBuf.reverse()

        loss = self.Net.loss(
            torch.from_numpy(np.vstack(stateBuf).astype(np.float32)).to(device),
            torch.from_numpy(np.vstack(actionBuf).astype(np.float32)).to(device),
            torch.from_numpy(np.array(RBuf).reshape(-1, 1).astype(np.float32)).to(device)
        )
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Net.parameters(), 40)
        self.opt.step()

        return loss

def main():
    a2c = A2C(lr=1e-5, maxEps=10000, maxSteps=200, updateStride=1, gamma=0.9, hiddenSize=256)
    a2c.train()
    # test()

if __name__ == '__main__':
    main()