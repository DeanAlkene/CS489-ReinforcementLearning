import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import random
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

Exp = namedtuple('Exp', ('state', 'action', 'reward', 'nextState', 'done'))

class Enviroment:
    def __init__(self, maxSteps):
        self.env = gym.make('MountainCar-v0').unwrapped
        self.max_episode_steps = 500

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
            reward = self.height(state[0]) + state[1]
            if state[0] > self.env.goal_position - 0.1:
                reward += 100 * (state[0] - self.env.goal_position + 0.1)
        return state, reward, done, info
    
    def height(self, xs):
        return np.sin(3 * xs)*.45+.55

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

    def test(self):
        for i in range(10):
            observation = self.env.reset()
            t = 0
            done = False
            while not done:
                self.env.render()
                action = self.env.action_space.sample()
                observation, reward, done, _ = self.env.step(action)
                if done:
                    print(observation)
                    print("Episode %d ended after %d timesteps" % (i + 1, t + 1))
                    break
                t += 1
        self.env.close()

class Agent:
    def __init__(self, actionSpace, observationSpace, gamma, epsilon, batchSize, lr, hiddenSize, updateStride):
        self.actionSpace = actionSpace
        self.actionNum = actionSpace.n
        self.actionSpace = [i for i in range(self.actionNum)]
        self.stateSize = observationSpace.shape[0]
        self.gamma = gamma
        self.epsilon = epsilon
        self.batchSize = batchSize
        self.lr = lr
        self.hiddenSize = hiddenSize
        self.updateStride = updateStride
        self.net = DQN(self.stateSize, self.hiddenSize, self.actionNum)
        self.targetNet = DQN(self.stateSize, self.hiddenSize, self.actionNum)
        self.optimizer = optim.Adam(params=self.net.parameters(), lr=self.lr)

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

        Q = self.net(stateBatch).gather(1, actionBatch)  # get Q(s, a)
        targetQ = self.targetNet(nextStateBatch).max(1)[0].view(-1, 1)
        y = (targetQ * self.gamma) * doneBatch + rewardBatch
        loss = F.mse_loss(Q, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def DeepQLearning(self, env, buf, episodeNum):
        totalN = 0
        scores = []
        steps = []
        self.targetNet.load_state_dict(self.net.state_dict())
        for i in range(episodeNum):
            score = 0.0
            state = env.reset()
            for t in count():
                #env.render()
                action = self.getAction(torch.tensor(state.reshape(1, self.stateSize), dtype=torch.float))
                nextState, reward, done, _ = env.step(action.item())
                buf.push(torch.tensor(state.reshape(1, self.stateSize), dtype=torch.float),
                          action,
                          torch.tensor([[reward]], dtype=torch.float),
                          torch.tensor(nextState.reshape(1, self.stateSize), dtype=torch.float),
                          torch.tensor([[not done]], dtype=torch.long))
                state = nextState
                score += reward
                self.optimzeDQN(buf)
                totalN += 1
                if totalN % self.updateStride == 0:
                    self.targetNet.load_state_dict(self.net.state_dict())
                if done or t + 1 >= env.max_episode_steps:
                    scores.append(score)
                    steps.append(t + 1)
                    print("Episode %d ended after %d timesteps with score %f" % (i + 1, t + 1, score))
                    break

        ep_range = [i + 1 for i in range(episodeNum)]
        plt.figure()
        plt.plot(ep_range, scores)
        plt.title('MountainCar')
        plt.xlabel('episode')
        plt.ylabel('score')
        plt.savefig('score')
        plt.figure()
        plt.plot(ep_range, steps)
        plt.title('MountainCar')
        plt.xlabel('episode')
        plt.ylabel('step')
        plt.savefig('step')

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
        stateSize = env.getObservationSpace().shape[0]
        actionNum = env.getActionSpace().n
        if initLen > self.capacity:
            return
        while len(self.buffer) < initLen:
            state = env.reset()
            for t in count():
                action = env.getActionSpace().sample()
                nextState, reward, done, _ = env.step(action)
                self.push(torch.tensor(state.reshape(1, stateSize), dtype=torch.float),
                          torch.tensor([[action]], dtype=torch.long),
                          torch.tensor([[reward]], dtype=torch.float),
                          torch.tensor(nextState.reshape(1, stateSize), dtype=torch.float),
                          torch.tensor([[not done]], dtype=torch.long))
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
        # x = F.sigmoid(self.fc1(x))
        # x = F.sigmoid(self.fc2(x))
        # x = F.sigmoid(self.fc3(x))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return x

def main():
    env = Enviroment(maxSteps=1000)
    buf = ReplayBuffer(capacity=10000)
    buf.fill(env, initLen=1000)
    agt = Agent(env.getActionSpace(),
                env.getObservationSpace(),
                gamma=0.99,
                epsilon=0.2,
                batchSize=128,
                lr=0.001,
                hiddenSize=256,
                updateStride=5)
    agt.DeepQLearning(env, buf, episodeNum=1000)
    # env.test()

if __name__ == "__main__":
    main()

# env = gym.make('MountainCar-v0')
# env._max_episode_steps = 500
# for i in range(10):
#     observation = env.reset()
#     t = 0
#     done = False
#     while not done:
#         env.render()
#         action = env.action_space.sample()
#         observation, reward, done, _ = env.step(action)
#         if done:
#             print(observation)
#             print("Episode %d ended after %d timesteps" % (i + 1, t + 1))
#             break
#         t += 1
# env.close()
