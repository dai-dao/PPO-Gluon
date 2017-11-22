import numpy as np 
import mxnet as mx 
from mxnet import gluon, autograd, nd
from utils import * 
import mxnet.ndarray as F


class ActorCritic_Discrete(gluon.Block):
    '''
        No activation method on the action logits, as the sampling, log_prob and entropy 
            from OpenAI seems to work for any scale
    '''

    def __init__(self, state_dim, action_dim, args, **kwargs):
        super(ActorCritic_Discrete, self).__init__(**kwargs)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.args = args 

        '''
        with self.name_scope():
            self.l1 = gluon.nn.Dense(in_units=self.state_dim, units=100, activation='relu')
            self.value = gluon.nn.Dense(in_units=100, units=1)
            self.logits = gluon.nn.Dense(in_units=100, units=action_dim)
        '''

        with self.name_scope():
            self.conv1 = gluon.nn.Conv2D(channels=32, kernel_size=8, strides=4, use_bias=True, activation='relu')
            self.conv2 = gluon.nn.Conv2D(channels=64, kernel_size=4, strides=2, use_bias=True, activation='relu')
            self.conv3 = gluon.nn.Conv2D(channels=64, kernel_size=3, strides=1, use_bias=True, activation='relu')
            self.l1 = gluon.nn.Dense(units=512, activation='relu')
            self.logits = gluon.nn.Dense(units=self.action_dim)
            self.value = gluon.nn.Dense(units=1)

        self.conv1.collect_params().initialize(OrthoInit(np.sqrt(2)), ctx=self.args.ctx)
        self.conv2.collect_params().initialize(OrthoInit(np.sqrt(2)), ctx=self.args.ctx)
        self.conv3.collect_params().initialize(OrthoInit(np.sqrt(2)), ctx=self.args.ctx)
        self.l1.collect_params().initialize(OrthoInit(np.sqrt(2)), ctx=self.args.ctx)
        self.logits.collect_params().initialize(OrthoInit(0.01), ctx=self.args.ctx)
        self.value.collect_params().initialize(OrthoInit(1.0), ctx=self.args.ctx)

        self.act_loss = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=True, from_logits=False)


    def forward(self, x):
        x = x / 255.

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.l1(x)

        value = self.value(x)
        logits = F.softmax(self.logits(x))
        return value, logits


    def get_value(self, x):
        value, _ = self(x)
        return value


    def choose_action(self, x):
        _, logits = self(x)
        action = self.sample(logits)
        return action 

    
    def sample(self, logits):
        # u = nd.random.uniform(shape=logits.shape)
        # return nd.argmax(logits - nd.log(-nd.log(u)), axis=-1)
        return nd.sample_multinomial(logits)


    def log_prob(self, logits, action):
        '''
            action : action index
            logits : unnormalized 
        '''
        # One number, Not vector output
        # This doesn't work
        return -self.act_loss(logits, action)


    def entropy(self, logits):
        # This works
        out = -nd.sum(logits * nd.log(logits + 1e-8), axis=1)
        return out


class ActorCritic_Gaussian(gluon.Block):
    def __init__(self, state_dim, action_dim, action_bound, args, **kwargs):
        super(ActorCritic_Gaussian, self).__init__(**kwargs)
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
        return nd.sum(nd.log(sigma + 1e-8) + .5 * np.log(2.0 * np.pi * np.e), axis=-1)


    def log_prob(self, x, mu, sigma):   
        return -0.5 * np.log(2.0 * np.pi) - nd.log(sigma + 1e-8) - (x - mu) ** 2 / (2 * sigma ** 2 + 1e-8)