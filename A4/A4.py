import numpy as np
import matplotlib.pyplot as plt
import gym
import torch

env = gym.make('MountainCar-v0')
print("action space")
print(env.action_space)
print("observation space")
print(env.observation_space)
print("High")
print(env.observation_space.high)
print("Low")
print(env.observation_space.low)
for i in range(10):
    observation = env.reset()
    t = 0
    done = False
    while not done:
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode %d ended after %d timesteps" % (i + 1, t + 1))
            break
        t += 1
env.close()
