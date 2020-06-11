import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import namedtuple, deque
from itertools import count
import random
import math
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image
import time
import cv2
cv2.ocl.setUseOpenCL(False)

Exp = namedtuple('Exp', ('state', 'action', 'reward', 'nextState', 'done'))
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resize = T.Compose([T.ToPILImage(),
                    T.ToTensor()])

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0

class LazyFrames(object):
    def __init__(self, frames):
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]

def getEnv(env_id):
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = WarpFrame(env)
    # env = ScaledFloatFrame(env)
    env = FrameStack(env, 4)
    return env

class Agent:
    def __init__(self, actionSpace, observationSpace, gamma, epsilon, tau, batchSize, lr, hiddenSize, learnStride, updateStride, maxStep, isDueling=False):
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
        self.learnStride = learnStride
        self.updateStride = updateStride
        self.maxStep = maxStep
        # self.net = DQN(self.stateSize[0], self.stateSize[1], self.stateSize[2], self.hiddenSize, self.actionNum).to(device)
        # self.targetNet = DQN(self.stateSize[0], self.stateSize[1], self.stateSize[2], self.hiddenSize, self.actionNum).to(device)
        if isDueling:
            self.net = DuelingDQN(84, 84, 4, self.hiddenSize, self.actionNum).to(device)
            self.targetNet = DuelingDQN(84, 84, 4, self.hiddenSize, self.actionNum).to(device)
        else:
            self.net = DQN(84, 84, 4, self.hiddenSize, self.actionNum).to(device)
            self.targetNet = DQN(84, 84, 4, self.hiddenSize, self.actionNum).to(device)
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
        self.epsilon -= (self.EPS_START - self.EPS_END) / self.EPS_DECAY
        self.epsilon = max(self.epsilon, self.EPS_END)

    def learn(self, buf, algo):
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

        if algo == 'DQN':
            Q = self.net(stateBatch).gather(1, actionBatch)  # get Q(s_t, a)
            targetQ = self.targetNet(nextStateBatch).max(1)[0].view(-1, 1) # max_a Q'(s_t+1, a)
            y = (targetQ * self.gamma) * doneBatch + rewardBatch
            loss = F.mse_loss(Q, y)
            estQMean = np.mean(targetQ.detach().cpu().numpy())
        elif algo == 'DDQN':
            Q = self.net(stateBatch).gather(1, actionBatch)  # get Q(s, a)
            targetAction = self.net(nextStateBatch).max(1)[1].view(-1, 1)  # get argmax Q'(s_t+1, a)
            targetQ = self.targetNet(nextStateBatch).gather(1, targetAction)
            y = (targetQ * self.gamma) * doneBatch + rewardBatch
            loss = F.mse_loss(Q, y)
            estQMean = np.mean(targetQ.detach().cpu().numpy())
        else:
            assert(0)

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
            state = np.array(env.reset())
            for t in count():
                env.render()
                action = self.getAction(resize(state).unsqueeze(0).to(device))
                nextState, reward, done, _ = env.step(action.item())
                nextState = np.array(nextState)
                buf.push(resize(state).unsqueeze(0).to("cpu"),
                          action.to("cpu"),
                          torch.tensor(np.array([reward]).reshape(1, -1), device="cpu", dtype=torch.float),
                          resize(nextState).unsqueeze(0).to("cpu"),
                          torch.tensor(np.array([not done]).reshape(1, -1), device="cpu", dtype=torch.long))
                state = nextState
                # score += rewardDecay * reward
                # rewardDecay *= self.gamma
                score += reward
                totalN += 1
                if totalN % self.learnStride == 0:
                    loss, mean = self.learn(buf, algo)
                    losses.append(loss)
                    Qmeans.append(mean)
                if ifDecay:
                    self.epsilonDecay(totalN)
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
            state = np.array(env.reset())
            for t in count():
                action = env.action_space.sample()
                nextState, reward, done, _ = env.step(action)
                nextState = np.array(nextState)
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

class DuelingDQN(nn.Module):
    def __init__(self, h, w, inputChannel, hiddenSize, outputSize):
        super(DuelingDQN, self).__init__()
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
        self.fc4_adv = nn.Linear(linear_input_size, hiddenSize)
        self.fc4_val = nn.Linear(linear_input_size, hiddenSize)
        self.head_adv = nn.Linear(hiddenSize, outputSize)
        self.head_val = nn.Linear(hiddenSize, 1)

        self.outputSize = outputSize

    def forward(self, x):
        x = x / 255
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        adv = F.leaky_relu(self.fc4_adv(x))
        val = F.leaky_relu(self.fc4_val(x))
        adv = self.head_adv(adv)
        val = self.head_val(val).expand(x.size(0), self.outputSize)
        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.outputSize)
        return x

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
    env = getEnv("BreakoutNoFrameskip-v4")
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
                learnStride=4,
                updateStride=10000,
                maxStep=1000,
                isDueling=True)
    agt.DeepQLearning(env, buf, episodeNum=20000, algo='DDQN', ifDecay=True)
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