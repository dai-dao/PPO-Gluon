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

        self.net = ActorCritic(self.observation_dim, self.action_dim, self.action_bound, self.args)
        # self.old_net = ActorCritic(self.observation_dim, self.action_dim, self.action_bound, self.args)

        self.net.collect_params().initialize(ctx=self.args.ctx)
        # self.old_net.collect_params().initialize(ctx=self.args.ctx)

        # Copy params from new to old
        # soft_update(self.old_net, self.net)
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

        value = self.get_value(s)
        action = self.choose_action(s)

        action_nd = nd.array(action, ctx=self.args.ctx).reshape((-1, self.action_dim))
        _, mu, sigma = self.net(s)
        logpac = self.net.log_gaussian(action_nd, mu, sigma)

        logpac = logpac.asnumpy()[0]

        return value, action, logpac


    def get_value(self, s):
        s = nd.array(s, ctx=self.args.ctx)
        s = nd.reshape(s, (-1, self.observation_dim))

        value = self.net.get_value(s)
        value = value.asnumpy()[0][0]     
        return value


    def update(self, b_s, b_a, b_r, b_logpac):
        b_s = nd.array(b_s, ctx=self.args.ctx).reshape((-1, self.observation_dim))
        b_a = nd.array(b_a, ctx=self.args.ctx).reshape((-1, self.action_dim))
        b_r = nd.array(b_r, ctx=self.args.ctx).reshape((-1, 1))
        b_oldpi_log_prob = nd.array(b_logpac, ctx=self.args.ctx).reshape((-1, self.action_dim))

        with autograd.record():
            # Value loss
            v_pred, mu, sigma = self.net(b_s)
            advantage = b_r - v_pred
            vf_loss = nd.mean(nd.square(advantage))

            # Detach from the computation graph
            advantage = advantage.detach()

            # Action loss
            pi_log_prob = self.net.log_gaussian(b_a, mu, sigma)
            # ratio = nd.exp(pi_log_prob - oldpi_log_prob)
            ratio = nd.exp(pi_log_prob - b_oldpi_log_prob)
            surr1 = ratio * advantage
            surr2 = nd.clip(ratio, 1.0 - self.args.clip_param, 1.0 + self.args.clip_param) * advantage
            actor_loss = -nd.mean(nd.minimum(surr1, surr2))
            entropy = self.net.entropy(sigma)

            # Total (maximize entropy to encourage exploration)
            loss = vf_loss * self.args.value_coefficient + actor_loss \
                    - entropy * self.args.entropy_coefficient

        loss.backward()
        self.trainer.step(b_s.shape[0])

    
    def new_update(self, obs, actions, values, dones, logpacs, returns, discounted_r, clip_param, lr):
        # advantages = returns - values 
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # advantages = nd.array(advantages, ctx=self.args.ctx).reshape((-1, 1))
        obs = nd.array(obs, ctx=self.args.ctx).reshape((-1, self.observation_dim))
        actions = nd.array(actions, ctx=self.args.ctx).reshape((-1, self.action_dim))
        values = nd.array(values, ctx=self.args.ctx).reshape((-1, 1))
        returns = nd.array(returns, ctx=self.args.ctx).reshape((-1, 1))
        oldpi_log_prob = nd.array(logpacs, ctx=self.args.ctx).reshape((-1, self.action_dim))
        discounted_r = nd.array(discounted_r, ctx=self.args.ctx).reshape((-1, 1))

        # Learning rate scheduling
        # self.trainer.set_learning_rate(lr)

        # Auto grad
        with autograd.record():
            # Value loss
            vpred, mu, sigma = self.net(obs)

            '''
            vpred_clipped = values + nd.clip(vpred - values, -clip_param, clip_param)
            vf_loss1 = nd.square(vpred - returns)
            vf_loss2 = nd.square(vpred_clipped - returns)
            vf_loss = nd.mean(nd.maximum(vf_loss1, vf_loss2))
            '''

            advantages = discounted_r - vpred
            vf_loss = nd.mean(nd.square(advantages)) # TESTING
            clip_param = 0.2 # TESTING

            advantages = advantages.detach()

            # Action loss
            pi_log_prob = self.net.log_gaussian(actions, mu, sigma)
            ratio = nd.exp(pi_log_prob - oldpi_log_prob)
            surr1 = ratio * advantages
            surr2 = nd.clip(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
            actor_loss = -nd.mean(nd.minimum(surr1, surr2))
            
            # Entropy term
            entropy = self.net.entropy(sigma)

            # Total loss
            loss = vf_loss * self.args.value_coefficient + actor_loss \
                        - entropy * self.args.entropy_coefficient

        # Compute gradients and updates
        loss.backward()
        self.trainer.step(obs.shape[0])