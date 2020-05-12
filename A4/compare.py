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

def compareDraw(filterSize):
    dqn_filtered = []
    ddqn_filtered = []
    dqn_filtered = np.load('DQN_Qmean.npy')
    ddqn_filtered = np.load('DDQN_Qmean.npy')
    ep_range1 = [i+1 for i in range(len(dqn_filtered))]
    ep_range2 = [i+1 for i in range(len(ddqn_filtered))]

    # for i in range(0, len(dqn), filterSize):
    #     dqn_filtered.append(np.mean(dqn[i: i + filterSize]))
    #     ep_range1.append(i + filterSize / 2)

    # for i in range(0, len(ddqn), filterSize):
    #     ddqn_filtered.append(np.mean(ddqn[i: i + filterSize]))
    #     ep_range2.append(i + filterSize / 2)

    plt.figure(figsize=(18, 9))
    plt.plot(ep_range1, dqn_filtered, label='DQN', color=plt.cm.tab10(0), alpha=0.6)
    plt.plot(ep_range2, ddqn_filtered, label='DDQN', color=plt.cm.tab10(1), alpha=0.6)
    plt.plot([ep_range1[0], ep_range1[len(ep_range1) - 1]], [778.2541514061293, 778.2541514061293], label='DQN true value', color=plt.cm.tab10(0), linewidth=2)
    plt.plot([ep_range2[0], ep_range2[len(ep_range2) - 1]], [790.9329366115888, 790.9329366115888], label='DDQN true value', color=plt.cm.tab10(1), linewidth=2)
    plt.title('MountainCar, DQN vs DDQN')
    plt.xlabel('step')
    plt.legend()
    plt.ylabel('Q')
    plt.savefig('compare')

def main():
    compareDraw(100)

if __name__ == "__main__":
    main()
