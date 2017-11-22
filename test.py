from model import *
import gym 
from mxnet import nd, gluon
import mxnet as mx
import numpy as np



'''
env = gym.make('CartPole-v0')

# Simple state / action space
action_dim = env.action_space.n
observation_dim = env.observation_space.shape[0]

# Network definition
net = ActorCritic(observation_dim, action_dim)
net.collect_params().initialize()

test_input = nd.uniform(shape=(1, observation_dim))
test_out = net(test_input)

print("Success forward")
'''


env = gym.make('CartPole-v0')
ob = env.reset()

print('Action space', env.action_space)
print('Observation space', env.observation_space)


def run_episode(env, parameters):  
    observation = env.reset()
    totalreward = 0
    for _ in range(200):
        action = 0 if np.matmul(parameters,observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        env.render()
        totalreward += reward
        if done:
            print('Episode ended | Reward is', totalreward)
            break
    return totalreward


bestparams = None  
bestreward = 0  
for _ in range(10000):  
    parameters = np.random.rand(4) * 2 - 1
    reward = run_episode(env,parameters)
    if reward > bestreward:
        bestreward = reward
        bestparams = parameters
        # considered solved if the agent lasts 200 timesteps
        if reward == 200:
            break