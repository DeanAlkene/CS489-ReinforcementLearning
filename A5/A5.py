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

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
# try:
#     mp.set_start_method('spawn')
# except RuntimeError:
#     pass

class SharedAdam(optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

class ACNet(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(ACNet, self).__init__()
        self.actor1 = nn.Linear(inputSize, hiddenSize)
        self.mu = nn.Linear(hiddenSize, outputSize)
        self.sigma = nn.Linear(hiddenSize, outputSize)

        self.critic1 = nn.Linear(inputSize, hiddenSize)
        self.value = nn.Linear(hiddenSize, 1)

        for l in [self.actor1, self.mu, self.sigma, self.critic1, self.value]:
            self._initLayer(l)

        self.distribution = torch.distributions.Normal

    def _initLayer(self, layer):
        nn.init.normal_(layer.weight, mean=0.0, std=0.1)
        nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        actor1 = F.relu6(self.actor1(x))
        mu = 2 * torch.tanh(self.mu(actor1))
        sigma = F.softplus(self.sigma(actor1)) + 0.001
        critic1 = F.relu6(self.critic1(x))
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
        actor_loss = -(log_prob * error.detach() + 0.005 * entropy)
        return (critic_loss + actor_loss).mean()

    def selectAction(self, state):
        self.training = False
        with torch.no_grad():
            mu, sigma, _ = self.forward(state)
            dist = self.distribution(mu.detach(), sigma.detach())
            return dist.sample().numpy()

class Worker(mp.Process):
    def __init__(self, rank, globalNet, localNet, optimizer, totalEpisode, globalReturn, Q, params):
        super(Worker, self).__init__()
        self.rank = rank
        self.env = gym.make('Pendulum-v0').unwrapped
        self.GNet = globalNet
        self.LNet = localNet
        self.opt = optimizer
        self.totEps = totalEpisode
        self.totR = globalReturn
        self.Q = Q
        self.params = params

    def run(self):
        steps = 1
        while self.totEps.value < self.params['MAX_EPISODE']:
            stateBuf, actionBuf, rewardBuf = [], [], []
            state = self.env.reset()
            ret = 0.0
            # rewardDecay = 1.0

            for t in range(self.params['MAX_STEP']):
                if self.rank == 0:
                    self.env.render()
                action = self.LNet.selectAction(torch.from_numpy(state.reshape(1, -1).astype(np.float32)).to(device))
                nextState, reward, done, _ = self.env.step(action.clip(-2, 2))
                if t == self.params['MAX_STEP'] - 1:
                    done = True
                ret += reward
                # ret += rewardDecay * reward
                # rewardDecay *= self.params['gamma']
                stateBuf.append(state.reshape(-1))
                actionBuf.append(action)
                rewardBuf.append((reward + 8.1) / 8.1)

                if steps % self.params['UPDATE_STRIDE'] == 0 or done:
                    loss = self.learn(nextState, done, stateBuf, actionBuf, rewardBuf)
                    stateBuf, actionBuf, rewardBuf = [], [], []

                    if done:
                        with self.totEps.get_lock():
                            self.totEps.value += 1
                        with self.totR.get_lock():
                            if self.totR.value == 0:
                                self.totR.value = ret
                            else:
                                self.totR.value = self.totR.value * 0.9 + ret * 0.1
                        self.Q.put(self.totR.value)
                        print("Rank: %d\tEps: %d\tTotRet: %f\tLoss: %f"%(self.rank, self.totEps.value, self.totR.value, loss))
                        break
                
                state = nextState
                steps += 1
        self.Q.put(None)
        self.env.close()

    def learn(self, nextState, done, stateBuf, actionBuf, rewardBuf):
        if done:
            R = 0
        else:
            R = self.LNet.forward(torch.from_numpy(nextState.reshape(1, -1).astype(np.float32)).to(device))[-1][0].item()
        
        RBuf = []
        for r in rewardBuf[::-1]:
            R = r + self.params['gamma'] * R
            RBuf.append(R)
        RBuf.reverse()

        loss = self.LNet.loss(
            torch.from_numpy(np.vstack(stateBuf).astype(np.float32)).to(device),
            torch.from_numpy(np.vstack(actionBuf).astype(np.float32)).to(device),
            torch.from_numpy(np.array(RBuf).reshape(-1, 1).astype(np.float32)).to(device)
        )
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.LNet.parameters(), 20)
        for l, g in zip(self.LNet.parameters(), self.GNet.parameters()):
            g._grad = l.grad
        self.opt.step()

        self.LNet.load_state_dict(self.GNet.state_dict())

        return loss
class A3C:
    def __init__(self, gamma, updateStride, maxEps, maxSteps, hiddenSize, lr):
        self.params = {'MAX_EPISODE': maxEps, 'MAX_STEP': maxSteps, 'UPDATE_STRIDE': updateStride, 'gamma': gamma, 'hiddenSize': hiddenSize}
        self.env = gym.make('Pendulum-v0').unwrapped
        self.globalNet = ACNet(self.env.observation_space.shape[0], hiddenSize, self.env.action_space.shape[0]).to(device)
        self.globalNet.share_memory()
        self.opt = SharedAdam(self.globalNet.parameters(), lr=lr, betas=(0.95, 0.999))
        self.totEps = mp.Value('i', 0)
        self.totR = mp.Value('d', 0.0)
        self.Q = mp.Queue()
    
    def train(self):
        workers = [Worker(rank, self.globalNet,
                          ACNet(self.env.observation_space.shape[0], self.params['hiddenSize'], self.env.action_space.shape[0]).to(device),
                          self.opt,
                          self.totEps,
                          self.totR,
                          self.Q,
                          self.params)
                          for rank in range(mp.cpu_count())]
        
        [w.start() for w in workers]
        res = []
        while True:
            r = self.Q.get()
            if r is not None:
                res.append(r)
            else:
                break
        [w.join() for w in workers]

        np.save('res', res)
        plt.plot(res)
        plt.ylabel('Average Return')
        plt.xlabel('Episodes')
        plt.savefig('res.png')
        torch.save(self.globalNet.state_dict(), 'net.pkl')
        self.env.close()

def test():
    env = gym.make('Pendulum-v0').unwrapped
    net = ACNet(env.observation_space.shape[0], 256, env.action_space.shape[0])
    net.load_state_dict(torch.load('net.pkl'))
    state = env.reset()
    while True:
        env.render()
        action = net.selectAction(torch.from_numpy(state.reshape(1, -1).astype(np.float32)).to(device))
        nextState, reward, done, _ = env.step(action.clip(-2, 2))
        state = nextState

def main():
    a3c = A3C(gamma=0.9,
              updateStride=5,
              maxEps=10000,
              maxSteps=200,
              hiddenSize=256,
              lr=1e-4)
    a3c.train()
    test()

if __name__ == '__main__':
    main()
