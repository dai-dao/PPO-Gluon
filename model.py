import numpy as np 
import mxnet as mx 
from mxnet import gluon, autograd, nd
from utils import * 


EPS = 0.003

# Try with 2 different networks this time
class Actor(gluon.Block):
    def __init__(self, state_dim, action_dim, action_bound, args, **kwargs):
        super(Actor, self).__init__(**kwargs)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.args = args

        with self.name_scope():
            self.l1 = gluon.nn.Dense(in_units=self.state_dim, units=256, activation='relu')
            self.l2 = gluon.nn.Dense(in_units=256, units=128, activation='relu')
            self.l3 = gluon.nn.Dense(in_units=128, units=64, activation='relu')
            self.mu = gluon.nn.Dense(in_units=64, units=action_dim, activation='tanh')
            self.sigma = gluon.nn.Dense(in_units=64, units=action_dim, activation='softrelu')

        '''
        self.l1.initialize(FaninInit())
        self.sigma.initialize(FaninInit())
        self.mu.initialize(mx.init.Uniform(EPS))
        '''


    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        mu = self.mu(x) * 2 # MUST scale by the action
        sigma = self.sigma(x)
        return mu, sigma


class Critic(gluon.Block):
    def __init__(self, state_dim, action_dim, action_bound, args, **kwargs):
        super(Critic, self).__init__(**kwargs)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.args = args

        with self.name_scope():
            self.l1 = gluon.nn.Dense(in_units=self.state_dim, units=256, activation='relu')
            self.l2 = gluon.nn.Dense(in_units=256, units=128, activation='relu')
            self.v = gluon.nn.Dense(in_units=128, units=1)
        
        '''
        self.l1.initialize(FaninInit())
        self.v.initialize(mx.init.Uniform(EPS))
        '''


    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.v(x)
        return x


# Doesn't work -> WHY???
class ActorCritic(gluon.Block):
    def __init__(self, state_dim, action_dim, action_bound, args, **kwargs):
        super(ActorCritic, self).__init__(**kwargs)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.args = args

        with self.name_scope():
            self.l1 = gluon.nn.Dense(in_units=self.state_dim, units=256, activation='relu')
            self.l2 = gluon.nn.Dense(in_units=256, units=256, activation='relu')
            self.mu = gluon.nn.Dense(in_units=256, units=action_dim, activation='tanh')
            self.sigma = gluon.nn.Dense(in_units=256, units=action_dim)
            self.value = gluon.nn.Dense(in_units=256, units=1)


    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        mu = self.mu(x)
        sigma = self.softplus(self.sigma(x))
        val = self.value(x)
        return val, mu, sigma


    def choose_action(self, x):
        _, mu, sigma = self(x)
        out = self.sample(mu, sigma)
        out = nd.clip(out, self.action_bound[0], self.action_bound[1])
        return out


    def get_value(self, x):
        val, _, _ = self(x)
        return val


    def softplus(self, x):
        return nd.log(1. + nd.exp(x))


    def sample(self, mu, sigma):
        epsilon = nd.random_normal(shape=mu.shape, loc=0., scale=1., ctx=self.args.ctx)
        out = mu + sigma * epsilon
        return out


    def log_gaussian(self, x, mu, sigma):   
        return -0.5 * np.log(2.0 * np.pi) - nd.log(sigma) - (x - mu) ** 2 / (2 * sigma ** 2)