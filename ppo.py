import numpy as np 
import mxnet as mx 
from mxnet import gluon, autograd, nd
from model import *
from utils import * 


class PPO(object):
    def __init__(self, env, args):
        self.action_dim = env.action_space.shape[0]
        self.observation_dim = env.observation_space.shape[0]
        self.action_bound = [env.action_space.low[0], env.action_space.high[0]]
        self.env = env
        self.args = args

        # Exploration
        self.noise = OrnsteinUhlenbeckActionNoise(self.action_dim)

        self.net = ActorCritic(self.observation_dim, self.action_dim, self.action_bound, self.args)
        self.old_net = ActorCritic(self.observation_dim, self.action_dim, self.action_bound, self.args)

        self.net.collect_params().initialize(ctx=self.args.ctx)
        self.old_net.collect_params().initialize(ctx=self.args.ctx)

        # Copy params from new to old
        soft_update(self.old_net, self.net)
        self.trainer = gluon.Trainer(self.net.collect_params(), 'adam', 
                                        {'learning_rate' : self.args.actor_lr})


    def choose_action(self, s):
        s = nd.array(s, ctx=self.args.ctx)
        s = nd.reshape(s, (-1, self.observation_dim))

        action = self.net.choose_action(s)
        action = action.asnumpy()[0]
        # Add noise to aid exploration
        # The noise IS the problem
        # action = action + self.noise.sample() #* 2) # Scaled by max possible action
        # Leave clip on for now 
        action = np.clip(action, self.action_bound[0], self.action_bound[1])
        return action


    def step(self, s):
        s = nd.array(s, ctx=self.args.ctx)
        s = nd.reshape(s, (-1, self.observation_dim))

        value, mu, sigma = self.net(s)
        action = self.net.sample(mu, sigma)
        logpac = self.net.log_gaussian(action, mu, sigma)

        value = value.asnumpy()[0][0]

        action = action.asnumpy()[0][0]
        action = np.clip(action, self.action_bound[0], self.action_bound[1])

        logpac = logpac.asnumpy()[0][0]

        return value, action, logpac


    def get_value(self, s):
        s = nd.array(s, ctx=self.args.ctx)
        s = nd.reshape(s, (-1, self.observation_dim))

        value = self.net.get_value(s)
        value = value.asnumpy()[0][0]     
        return value


    def update(self, b_s, b_a, b_r):
        # Copy params from new to old
        soft_update(self.old_net, self.net)

        b_s = nd.array(b_s, ctx=self.args.ctx).reshape((-1, self.observation_dim))
        b_a = nd.array(b_a, ctx=self.args.ctx).reshape((-1, self.action_dim))
        b_r = nd.array(b_r, ctx=self.args.ctx).reshape((-1, 1))

        _, old_mu, old_sigma = self.old_net(b_s)
        oldpi_log_prob = self.old_net.log_gaussian(b_a, old_mu, old_sigma)
        
        for _ in range(self.args.num_update_steps):
            with autograd.record():
                # Value loss
                v_pred, mu, sigma = self.net(b_s)
                advantage = b_r - v_pred
                vf_loss = nd.mean(nd.square(advantage))

                # Detach from the computation graph
                advantage = advantage.detach()

                # Action loss
                pi_log_prob = self.net.log_gaussian(b_a, mu, sigma)
                ratio = nd.exp(pi_log_prob - oldpi_log_prob)
                surr1 = ratio * advantage
                surr2 = nd.clip(ratio, 1.0 - self.args.clip_param, 1.0 + self.args.clip_param) * advantage
                actor_loss = -nd.mean(nd.minimum(surr1, surr2))
                entropy = self.net.entropy(sigma)

                # Total (maximize entropy to encourage exploration)
                loss = vf_loss * self.args.value_coefficient + actor_loss \
                        - entropy * self.args.entropy_coefficient

            loss.backward()
            self.trainer.step(b_s.shape[0])