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

class Agent:
    def __init__(self, actionSpace, observationSpace, gamma, epsilon, tau, batchSize, lr, hiddenSize, updateStride, maxStep):
        self.actionSpace = actionSpace
        self.actionNum = actionSpace.n
        self.actionSpace = [i for i in range(self.actionNum)]
        self.stateSize = observationSpace.shape[0]
        self.gamma = gamma
        self.epsilon = epsilon[0]
        self.EPS_START = epsilon[0]
        self.EPS_END = epsilon[1]
        self.EPS_DECAY = epsilon[2]
        self.tau = tau
        self.batchSize = batchSize
        self.lr = lr
        self.hiddenSize = hiddenSize
        self.updateStride = updateStride
        self.maxStep = maxStep
        self.net = DQN(self.stateSize, self.hiddenSize, self.actionNum).to(device)
        self.targetNet = DQN(self.stateSize, self.hiddenSize, self.actionNum).to(device)
        self.optimizer = optim.Adam(params=self.net.parameters(), lr=self.lr)

    def getAction(self, state, ifEpsilonGreedy=True):
        if ifEpsilonGreedy:
            if random.random() > self.epsilon:
                with torch.no_grad():
                    return self.net(state).max(1)[1].view(1, 1)
            else:
                return torch.tensor([[random.randrange(self.actionNum)]], device=device, dtype=torch.long)
        else:
            with torch.no_grad():
                return self.net(state).max(1)[1].view(1, 1)
    
    def epsilonDecay(self, N):
        if N >= 50000:
            self.epsilon = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1.0 * (N - 50000) / self.EPS_DECAY)

    def optimzeDQN(self, buf):
        if len(buf) < self.batchSize:
            print("Can't fetch enough exp!")
            return
        exps = buf.sample(self.batchSize)
        batch = Exp(*zip(*exps))  # batch => Exp of batch
        stateBatch = torch.cat(batch.state) # batchSize * stateSpace.shape[0]
        actionBatch = torch.cat(batch.action) # batchSize * 1
        rewardBatch = torch.cat(batch.reward) # batchSize * 1
        nextStateBatch = torch.cat(batch.nextState) # batchSize * stateSpace.shape[0]
        doneBatch = torch.cat(batch.done) # batchSize * 1

        Q = self.net(stateBatch).gather(1, actionBatch)  # get Q(s_t, a)
        targetQ = self.targetNet(nextStateBatch).max(1)[0].view(-1, 1) # max_a Q'(s_t+1, a)
        y = (targetQ * self.gamma) * doneBatch + rewardBatch
        loss = F.mse_loss(Q, y)
        estQMean = np.mean(targetQ.detach().cpu().numpy())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), estQMean
    
    def optimzeDDQN(self, buf):
        if len(buf) < self.batchSize:
            print("Can't fetch enough exp!")
            return
        exps = buf.sample(self.batchSize)
        batch = Exp(*zip(*exps))  # batch => Exp of batch
        stateBatch = torch.cat(batch.state) # batchSize * stateSpace.shape[0]
        actionBatch = torch.cat(batch.action) # batchSize * 1
        rewardBatch = torch.cat(batch.reward) # batchSize * 1
        nextStateBatch = torch.cat(batch.nextState) # batchSize * stateSpace.shape[0]
        doneBatch = torch.cat(batch.done) # batchSize * 1

        Q = self.net(stateBatch).gather(1, actionBatch)  # get Q(s, a)
        targetAction = self.net(nextStateBatch).max(1)[1].view(-1, 1)  # get argmax Q'(s_t+1, a)
        targetQ = self.targetNet(nextStateBatch).gather(1, targetAction)
        y = (targetQ * self.gamma) * doneBatch + rewardBatch
        loss = F.mse_loss(Q, y)
        estQMean = np.mean(targetQ.detach().cpu().numpy())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), estQMean

    def DeepQLearning(self, env, buf, episodeNum, algo="DQN", ifDecay=True):
        totalN = 0
        scores = []
        steps = []
        losses = []
        Qmeans = []
        self.targetNet.load_state_dict(self.net.state_dict())
        for i in range(episodeNum):
            score = 0.0
            # rewardDecay = 1.0
            state = env.reset()
            for t in count():
                env.render()
                action = self.getAction(torch.tensor(state.unsqueeze(0), device=device, dtype=torch.float))
                nextState, reward, done, _ = env.step(action.item())
                buf.push(torch.tensor(state.unsqueeze(0), device=device, dtype=torch.float),
                          action,
                          torch.tensor([[reward]].reshape(1, -1), device=device, dtype=torch.float),
                          torch.tensor(nextState.unsqueeze(0), device=device, dtype=torch.float),
                          torch.tensor([[not done]].reshape(1, -1), device=device, dtype=torch.long))
                state = nextState
                # score += rewardDecay * reward
                # rewardDecay *= self.gamma
                score += reward
                if algo == "DQN":
                    loss, mean = self.optimzeDQN(buf)
                elif algo == "DDQN":
                    loss, mean = self.optimzeDDQN(buf)
                elif algo == "DDDQN":
                    print("Developing")
                else:
                    assert(0)
                losses.append(loss)
                Qmeans.append(mean)
                if ifDecay:
                    self.epsilonDecay(totalN)
                totalN += 1
                if totalN % self.updateStride == 0:
                    self.targetNet.load_state_dict(self.net.state_dict())
                if done or t + 1 >= self.maxStep:
                    scores.append(score)
                    steps.append(t + 1)
                    print("Episode %d ended after %d timesteps with score %f, epsilon=%f" % (i + 1, t + 1, score, self.epsilon))
                    break
        
        np.save('score', scores)
        np.save('step', steps)
        np.save('loss',losses)
        np.save('Qmean',Qmeans)
        torch.save(self.net.state_dict(), 'net.pkl')
        torch.save(self.targetNet.state_dict(), 'targetNet.pkl')

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
        stateSize = env.observation_space.shape
        if initLen > self.capacity:
            return
        while len(self.buffer) < initLen:
            state = env.reset()
            for t in count():
                action = env.action_space.sample()
                nextState, reward, done, _ = env.step(action)
                self.push(torch.tensor(state.unsqueeze(0), device=device, dtype=torch.float),
                          torch.tensor(action.reshape(1, -1), device=device, dtype=torch.float),
                          torch.tensor(np.array([reward]).reshape(1, -1), device=device, dtype=torch.float),
                          torch.tensor(nextState.unsqueeze(0), device=device, dtype=torch.float),
                          torch.tensor(np.array([not done]).reshape(1, -1), device=device, dtype=torch.long))
                state = nextState
                if done or t + 1 >= maxSteps:
                    break

class DQN(nn.Module):
    def __init__(self, h, w, inputChannel, hiddenSize, outputSize):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(inputChannel, 32, 8, 4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.bn3 = nn.BatchNorm2d(64)
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64
        self.fc4 = nn.Linear(linear_input_size, hiddenSize)
        self.head = nn.Linear(hiddenSize, outputSize)

    def forward(self, x):
        x = x.float() / 255
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.fc4(x.view(x.size(0), -1)))
        return self.head(x)

def testNet(env, method):
    net = DQN(2, 256, 3)
    net.load_state_dict(torch.load('net.pkl'))
    avg = 0.0
    for i in range(1000):
        state = env.reset()
        Q = 0.0
        for t in count():
            # env.render()
            action = net(torch.FloatTensor([[float(state[0]), float(state[1])]])).max(1)[1].view(-1)
            Q += net(torch.FloatTensor([[float(state[0]), float(state[1])]])).max(1)[0].view(-1).item()
            state, reward, done, _ = env.step(action.item())
            if done:
                print("%d, %f" % (t + 1, Q / (t + 1)))
                avg += Q / (t + 1)
                break
    print(avg / 1000)
    
def main():
    env = gym.make("MountainCar-v0")
    buf = ReplayBuffer(capacity=100000)
    buf.fill(env, initLen=1000)
    agt = Agent(env.getActionSpace(),
                env.getObservationSpace(),
                gamma=0.99,
                epsilon=[1.0, 0.01, 20000],
                tau=0.01,
                batchSize=128,
                lr=0.001,
                hiddenSize=256,
                updateStride=10,
                maxStep=1000)
    agt.DeepQLearning(env, buf, episodeNum=5000, ifDecay=True)
    # agt.DoubleDeepQLearning(env, buf, episodeNum=5000, ifDecay=True)
    # draw(5000, 'DDQN', '', 50)
    # testNet(env, 'DDQN')
    env.close()

if __name__ == "__main__":
    main()