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

class Enviroment:
    def __init__(self, maxSteps):
        self.env = gym.make('MountainCar-v0').unwrapped
        self.max_episode_steps = maxSteps

    def reset(self):
        return self.env.reset()

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        if done:
            reward = 1000
        else:
            reward = (self.height(state[0]) - 0.1) * 10 + 50 * state[1] * state[1]
        return state, reward, done, info
    
    def height(self, xs):
        return np.sin(3 * xs) * .45 + .55

    def getActionSpace(self):
        return self.env.action_space

    def getObservationSpace(self):
        return self.env.observation_space

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
    def __init__(self, actionSpace, observationSpace, gamma, epsilon, tau, batchSize, lr, hiddenSize, updateStride):
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
        if (N >= 50000):
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

    def DeepQLearning(self, env, buf, episodeNum, ifDecay=True):
        totalN = 0
        scores = []
        steps = []
        losses = []
        Qmeans = []
        self.targetNet.load_state_dict(self.net.state_dict())
        for i in range(episodeNum):
            score = 0.0
            rewardDecay = 1.0
            state = env.reset()
            for t in count():
                env.render()
                action = self.getAction(torch.tensor(state.reshape(1, self.stateSize), device=device, dtype=torch.float))
                nextState, reward, done, _ = env.step(action.item())
                buf.push(torch.tensor(state.reshape(1, self.stateSize), device=device, dtype=torch.float),
                          action,
                          torch.tensor([[reward]], device=device, dtype=torch.float),
                          torch.tensor(nextState.reshape(1, self.stateSize), device=device, dtype=torch.float),
                          torch.tensor([[not done]], device=device, dtype=torch.long))
                state = nextState
                score += rewardDecay * reward
                rewardDecay *= self.gamma
                loss, mean = self.optimzeDQN(buf)
                losses.append(loss)
                Qmeans.append(mean)
                if ifDecay:
                    self.epsilonDecay(totalN)
                totalN += 1
                if totalN % self.updateStride == 0:
                    self.targetNet.load_state_dict(self.net.state_dict())
                if done or t + 1 >= env.max_episode_steps:
                    scores.append(score)
                    steps.append(t + 1)
                    print("Episode %d ended after %d timesteps with score %f, epsilon=%f" % (i + 1, t + 1, score, self.epsilon))
                    break
        
        np.save('DQN_score', scores)
        np.save('DQN_step', steps)
        np.save('DQN_loss',losses)
        np.save('DQN_Qmean',Qmeans)
        torch.save(self.net.state_dict(), 'net.pkl')
        torch.save(self.targetNet.state_dict(), 'targetNet.pkl')

    def DoubleDeepQLearning(self, env, buf, episodeNum, ifDecay=True):
        totalN = 0
        scores = []
        steps = []
        losses = []
        Qmeans = []
        self.targetNet.load_state_dict(self.net.state_dict())
        for i in range(episodeNum):
            score = 0.0
            rewardDecay = 1.0
            state = env.reset()
            for t in count():
                env.render()
                action = self.getAction(torch.tensor(state.reshape(1, self.stateSize), device=device, dtype=torch.float))
                nextState, reward, done, _ = env.step(action.item())
                buf.push(torch.tensor(state.reshape(1, self.stateSize), device=device, dtype=torch.float),
                          action,
                          torch.tensor([[reward]], device=device, dtype=torch.float),
                          torch.tensor(nextState.reshape(1, self.stateSize), device=device, dtype=torch.float),
                          torch.tensor([[not done]], device=device, dtype=torch.long))
                state = nextState
                score += rewardDecay * reward
                rewardDecay *= self.gamma
                loss, mean = self.optimzeDDQN(buf)
                losses.append(loss)
                Qmeans.append(mean)
                if ifDecay:
                    self.epsilonDecay(totalN)
                totalN += 1
                if totalN % self.updateStride == 0:
                    self.targetNet.load_state_dict(self.net.state_dict())
                    # for targetParam, param in zip(self.targetNet.parameters(), self.net.parameters()):
                    #     targetParam.data.copy_(self.tau * targetParam + (1 - self.tau) * param)
                if done or t + 1 >= env.max_episode_steps:
                    scores.append(score)
                    steps.append(t + 1)
                    print("Episode %d ended after %d timesteps with score %f, epsilon=%f" % (i + 1, t + 1, score, self.epsilon))
                    break
        
        np.save('DDQN_score', scores)
        np.save('DDQN_step', steps)
        np.save('DDQN_loss', losses)
        np.save('DDQN_Qmean',Qmeans)
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
    
    def fill(self, env, initLen):
        stateSize = env.getObservationSpace().shape[0]
        actionNum = env.getActionSpace().n
        if initLen > self.capacity:
            return
        while len(self.buffer) < initLen:
            state = env.reset()
            for t in count():
                action = env.getActionSpace().sample()
                nextState, reward, done, _ = env.step(action)
                self.push(torch.tensor(state.reshape(1, stateSize), device=device, dtype=torch.float),
                          torch.tensor([[action]], device=device, dtype=torch.long),
                          torch.tensor([[reward]], device=device, dtype=torch.float),
                          torch.tensor(nextState.reshape(1, stateSize), device=device, dtype=torch.float),
                          torch.tensor([[not done]], device=device, dtype=torch.long))
                state = nextState
                if done or t + 1 >= env.max_episode_steps:
                    break

class DQN(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(inputSize, hiddenSize)
        self.fc2 = nn.Linear(hiddenSize, hiddenSize)
        self.fc3 = nn.Linear(hiddenSize, outputSize)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return x

def draw(episodeNum, methodName, suffix, filterSize):
    scores_filtered = []
    steps_filtered = []
    ep_range = [i + 1 for i in range(episodeNum)]
    loose_range = []
    scores = np.load(methodName + '_score.npy')
    steps = np.load(methodName + '_step.npy')
    losses = np.load(methodName + '_loss.npy')
    Qmeans = np.load(methodName + '_Qmean.npy')
    for i in range(0, episodeNum, filterSize):
        scores_filtered.append(np.mean(scores[i: i + filterSize]))
        steps_filtered.append(np.mean(steps[i: i + filterSize]))
        loose_range.append(i + filterSize / 2)
    np.save('smooth_score', scores_filtered)
    np.save('smooth_step', steps_filtered)
    plt.figure(figsize=(18, 9))
    plt.plot(ep_range, scores, alpha=0.8)
    plt.plot(loose_range, scores_filtered)
    plt.title('MountainCar')
    plt.xlabel('episode')
    plt.ylabel('score')
    plt.savefig('score')
    plt.figure(figsize=(18, 9))
    plt.plot(ep_range, steps, alpha=0.8)
    plt.plot(loose_range, steps_filtered)
    plt.title('MountainCar')
    plt.xlabel('episode')
    plt.ylabel('step')
    plt.savefig('step')
    plt.figure(figsize=(18, 9))
    plt.plot(losses)
    plt.title('MountainCar')
    plt.xlabel('time')
    plt.ylabel('loss')
    plt.savefig('loss')
    plt.figure(figsize=(18, 9))
    plt.plot(Qmeans)
    plt.title('MountainCar')
    plt.xlabel('time')
    plt.ylabel('Q')
    plt.savefig('Q')

    data = []
    net = DQN(2, 256, 3)
    net.load_state_dict(torch.load('net' + suffix + '.pkl'))
    for i in range(10000):
        data.append(torch.tensor([[np.random.uniform(-1.2, 0.6), np.random.uniform(-0.07, 0.07)]], dtype=torch.float))
    data = torch.cat(data)
    with torch.no_grad():
        label = net(data).max(1)[1].view(-1).numpy()
    data = pd.DataFrame(data.numpy(), columns=['x', 'y'])
    label = pd.DataFrame(label, columns=['label'])
    colors = [plt.cm.tab10(i / float(4.0)) for i in range(4)]
    data = pd.concat([data, label], axis=1)

    actionName = ['left', 'neutral', 'right']
    plt.figure()
    for i in range(3):
        plt.scatter(data.loc[data.label == i].x, data.loc[data.label == i].y, s=10, label=actionName[i], cmap=colors[i], alpha=0.5)
    plt.title('Policy')
    plt.legend()
    plt.xlabel('position')
    plt.ylabel('velocity')
    plt.savefig(methodName + '_policy')

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
    env = Enviroment(maxSteps=500)
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
                updateStride=10)
    agt.DeepQLearning(env, buf, episodeNum=5000, ifDecay=True)
    # agt.DoubleDeepQLearning(env, buf, episodeNum=5000, ifDecay=True)
    # draw(5000, 'DDQN', '', 50)
    # testNet(env, 'DDQN')
    env.close()

if __name__ == "__main__":
    main()