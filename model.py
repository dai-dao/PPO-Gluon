import numpy as np 
import mxnet as mx 
from mxnet import gluon, autograd, nd
from utils import * 


# Doesn't work -> WHY??? -> Scale the action
class ActorCritic(gluon.Block):
    def __init__(self, state_dim, action_dim, action_bound, args, **kwargs):
        super(ActorCritic, self).__init__(**kwargs)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.args = args

        with self.name_scope():
            self.l1 = gluon.nn.Dense(in_units=self.state_dim, units=100, activation='relu')
            self.mu = gluon.nn.Dense(in_units=100, units=action_dim, activation='tanh')
            self.sigma = gluon.nn.Dense(in_units=100, units=action_dim, activation='softrelu')
            self.value = gluon.nn.Dense(in_units=100, units=1)


    def forward(self, x):
        x = self.l1(x)
        mu = self.mu(x) * self.action_bound[1] # ALWAYS scale the mean by action
        sigma = self.sigma(x)
        val = self.value(x)
        return val, mu, sigma


    def choose_action(self, x):
        _, mu, sigma = self(x)
        out = self.sample(mu, sigma)
        return out


    def get_value(self, x):
        val, _, _ = self(x)
        return val


    def sample(self, mu, sigma):
        epsilon = nd.random_normal(shape=mu.shape, loc=0., scale=1., ctx=self.args.ctx)
        out = mu + sigma * epsilon
        return out


    def entropy(self, sigma):
        return nd.sum(nd.log(sigma + 1e-5) + .5 * np.log(2.0 * np.pi * np.e), axis=-1)


    def log_gaussian(self, x, mu, sigma):   
        return -0.5 * np.log(2.0 * np.pi) - nd.log(sigma + 1e-5) - (x - mu) ** 2 / (2 * sigma ** 2 + 1e-5)