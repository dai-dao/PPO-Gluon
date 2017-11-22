import numpy as np 
import mxnet as mx 
from mxnet import gluon, autograd, nd
from model import *
from utils import * 
import random


class PPO_Discrete(object):
    def __init__(self, env, args):
        self.action_dim = env.action_space.n
        self.observation_dim = env.observation_space.shape
        self.env = env
        self.args = args
        self.loss_names = ['policy_loss', 'value_loss'] #, 'policy_entropy']

        ''' For discrete control environment (CartPole, BreakOut) '''
        self.net = ActorCritic_Discrete(self.observation_dim, self.action_dim, self.args)

        ''' Initialize parameters and optimizer '''
        # self.net.collect_params().initialize(ctx=self.args.ctx)
        self.trainer = gluon.Trainer(self.net.collect_params(), 'adam', 
                                        {'learning_rate' : self.args.lr})
        self.net.collect_params().zero_grad()


    def step(self, s, num_steps_so_far):
        # s is channel-last, NHWC
        s = np.transpose(s, (0, 3, 1, 2))
        s = nd.array(s, ctx=self.args.ctx)

        value, logits = self.net(s)
        # action = nd.argmax(logits, axis=1)
        # action = self.net.sample(logits)
        # logpac = self.net.log_prob(logits, action)

        # Epsilon greedy exploration
        eps = np.maximum(1. - num_steps_so_far / self.args.annealing_end, self.args.epsilon_min)
        action = nd.empty(shape=s.shape[0], ctx=self.args.ctx)
        logits_np = logits.asnumpy()
        for i in range(s.shape[0]):
            sample = np.random.random()
            if sample < eps:
                ac = random.randint(0, self.action_dim - 1)
            else:
                ac = int(np.argmax(logits[i]))
            action[i] = ac

        # Pick the probability of the chosen action
        logpac = nd.pick(logits, action, 1)

        # Reshaping the output
        logpac = logpac.asnumpy().reshape((-1))
        action = action.asnumpy().astype(np.int32).reshape((-1))
        value = value.asnumpy().reshape((-1))

        return value, action, logpac


    def update(self, obs, returns, masks, actions, values, logpacs, lrnow, cliprange_now):
        advantages = returns - values 
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = nd.array(advantages, ctx=self.args.ctx) # .reshape((-1, 1))

        obs = np.transpose(obs, (0, 3, 1, 2))
        obs = nd.array(obs, ctx=self.args.ctx)
        actions = nd.array(actions, ctx=self.args.ctx).reshape((-1, 1))
        values = nd.array(values, ctx=self.args.ctx).reshape((-1, 1))
        returns = nd.array(returns, ctx=self.args.ctx).reshape((-1, 1))
        oldpi_log_prob = nd.array(logpacs, ctx=self.args.ctx).reshape((-1, 1))

        # self.trainer.set_learning_rate(lrnow)

        # Auto grad
        with autograd.record():
            # Value loss
            vpred, logits = self.net(obs)
            vpred_clipped = values + nd.clip(vpred - values, -cliprange_now, cliprange_now)
            vf_loss1 = nd.square(vpred - returns)
            vf_loss2 = nd.square(vpred_clipped - returns)
            vf_loss = nd.mean(nd.maximum(vf_loss1, vf_loss2))

            # Action loss
            # pi_log_prob = self.net.log_prob(logits, actions)
            pi_log_prob = nd.pick(logits, actions, 1)
            ratio = nd.exp(pi_log_prob - oldpi_log_prob)
            surr1 = ratio * advantages
            surr2 = nd.clip(ratio, 1.0 - cliprange_now, 1.0 + cliprange_now) * advantages
            actor_loss = -nd.mean(nd.minimum(surr1, surr2))
            
            # Entropy term
            # entropy = self.net.entropy(logits)

            # Total loss
            # loss = vf_loss * self.args.value_coefficient + actor_loss 
                         # - entropy * self.args.entropy_coefficient
            loss = vf_loss + actor_loss

        # Compute gradients and updates
        loss.backward()
        self.trainer.step(obs.shape[0])

        return actor_loss.asscalar(), vf_loss.asscalar() #, entropy.asscalar()
