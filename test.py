from model import ActorCritic
import gym 
from mxnet import nd, gluon
import mxnet as mx


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