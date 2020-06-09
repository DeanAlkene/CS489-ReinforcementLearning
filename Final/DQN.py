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
from torchvision import transforms as T
from PIL import Image
import time

Exp = namedtuple('Exp', ('state', 'action', 'reward', 'nextState', 'done'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resize = T.Compose([T.ToPILImage(),
                    T.Resize((84, 84), interpolation=Image.CUBIC),
                    T.ToTensor()])
# device = "cpu"

class Agent:
    def __init__(self, actionSpace, observationSpace, gamma, epsilon, tau, batchSize, lr, hiddenSize, updateStride, maxStep):
        self.actionSpace = actionSpace
        self.actionNum = actionSpace.n
        self.actionSpace = [i for i in range(self.actionNum)]
        self.stateSize = observationSpace.shape
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
        # self.net = DQN(self.stateSize[0], self.stateSize[1], self.stateSize[2], self.hiddenSize, self.actionNum).to(device)
        # self.targetNet = DQN(self.stateSize[0], self.stateSize[1], self.stateSize[2], self.hiddenSize, self.actionNum).to(device)
        self.net = DQN(84, 84, 3, self.hiddenSize, self.actionNum).to(device)
        self.targetNet = DQN(84, 84, 3, self.hiddenSize, self.actionNum).to(device)
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
        # if N >= 50000:
        self.epsilon = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1.0 * N / self.EPS_DECAY)

    def optimzeDQN(self, buf):
        if len(buf) < self.batchSize:
            print("Can't fetch enough exp!")
            return
        exps = buf.sample(self.batchSize)
        batch = Exp(*zip(*exps))  # batch => Exp of batch
        stateBatch = torch.cat(batch.state).to(device) # batchSize * stateSpace.shape[0]
        actionBatch = torch.cat(batch.action).to(device) # batchSize * 1
        rewardBatch = torch.cat(batch.reward).to(device) # batchSize * 1
        nextStateBatch = torch.cat(batch.nextState).to(device) # batchSize * stateSpace.shape[0]
        doneBatch = torch.cat(batch.done).to(device)  # batchSize * 1

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
        stateBatch = torch.cat(batch.state).to(device) # batchSize * stateSpace.shape[0]
        actionBatch = torch.cat(batch.action).to(device) # batchSize * 1
        rewardBatch = torch.cat(batch.reward).to(device) # batchSize * 1
        nextStateBatch = torch.cat(batch.nextState).to(device) # batchSize * stateSpace.shape[0]
        doneBatch = torch.cat(batch.done).to(device)  # batchSize * 1

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
            startTime = time.time()
            state = env.reset()
            for t in count():
                #env.render()
                action = self.getAction(resize(state).unsqueeze(0).to(device))
                nextState, reward, done, _ = env.step(action.item())
                buf.push(resize(state).unsqueeze(0).to("cpu"),
                          action.to("cpu"),
                          torch.tensor(np.array([reward]).reshape(1, -1), device="cpu", dtype=torch.float),
                          resize(nextState).unsqueeze(0).to("cpu"),
                          torch.tensor(np.array([not done]).reshape(1, -1), device="cpu", dtype=torch.long))
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
                # if done or t + 1 >= self.maxStep:
                if done:
                    scores.append(score)
                    steps.append(t + 1)
                    endingTime = time.time()
                    print("Episode %d ended after %d timesteps with score %f, epsilon=%f in %fs" % (i + 1, t + 1, score, self.epsilon, endingTime-startTime))
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
                self.push(resize(state).unsqueeze(0).to("cpu"),
                          torch.tensor(np.array([action]).reshape(1, -1), device="cpu", dtype=torch.long),
                          torch.tensor(np.array([reward]).reshape(1, -1), device="cpu", dtype=torch.float),
                          resize(nextState).unsqueeze(0).to("cpu"),
                          torch.tensor(np.array([not done]).reshape(1, -1), device="cpu", dtype=torch.long))
                state = nextState
                # if done or t + 1 >= maxSteps:
                if done:
                    break

class DQN(nn.Module):
    def __init__(self, h, w, inputChannel, hiddenSize, outputSize):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(inputChannel, 16, 8, 4)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3, 1)
        self.bn3 = nn.BatchNorm2d(32)
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 32
        self.fc4 = nn.Linear(linear_input_size, hiddenSize)
        self.head = nn.Linear(hiddenSize, outputSize)

    def forward(self, x):
        x = x / 255
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.fc4(x.view(x.size(0), -1)))
        return self.head(x)

# def testNet(env, method):
#     net = DQN(2, 256, 3)
#     net.load_state_dict(torch.load('net.pkl'))
#     avg = 0.0
#     for i in range(1000):
#         state = env.reset()
#         Q = 0.0
#         for t in count():
#             # env.render()
#             action = net(torch.FloatTensor([[float(state[0]), float(state[1])]])).max(1)[1].view(-1)
#             Q += net(torch.FloatTensor([[float(state[0]), float(state[1])]])).max(1)[0].view(-1).item()
#             state, reward, done, _ = env.step(action.item())
#             if done:
#                 print("%d, %f" % (t + 1, Q / (t + 1)))
#                 avg += Q / (t + 1)
#                 break
#     print(avg / 1000)
    
def main():
    env = gym.make("BreakoutNoFrameskip-v4")
    buf = ReplayBuffer(capacity=100000)
    buf.fill(env, initLen=1000, maxSteps=200)
    agt = Agent(env.action_space,
                env.observation_space,
                gamma=0.99,
                epsilon=[1.0, 0.1, 1000000],
                tau=0.01,
                batchSize=32,
                lr=1e-4,
                hiddenSize=256,
                updateStride=20,
                maxStep=1000)
    agt.DeepQLearning(env, buf, episodeNum=10000, ifDecay=True)
    # agt.DoubleDeepQLearning(env, buf, episodeNum=5000, ifDecay=True)
    # draw(5000, 'DDQN', '', 50)
    # testNet(env, 'DDQN')
    env.close()

if __name__ == "__main__":
    main()
    # env = gym.make("PongNoFrameskip-v4")
    # for i in range(10):
    #     state = env.reset()
    #     for t in count():
    #         env.render()
    #         action = env.action_space.sample()
    #         state, reward, done, _ = env.step(action)
    #         print(state)
    #         if done:
    #             print("%d Ended" % (t + 1))
    #             break