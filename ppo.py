import numpy as np 
import mxnet as mx 
from mxnet import gluon, autograd, nd
from model import *
from utils import * 


class PPO_Discrete(object):
    def __init__(self, env, args):
        self.action_dim = env.action_space.n
        self.observation_dim = env.observation_space.shape
        self.env = env
        self.args = args
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy']

        ''' For discrete control environment (CartPole, BreakOut) '''
        self.net = ActorCritic_Discrete(self.observation_dim, self.action_dim, self.args)

        ''' Initialize parameters and optimizer '''
        # self.net.collect_params().initialize(ctx=self.args.ctx)
        self.trainer = gluon.Trainer(self.net.collect_params(), 'adam', 
                                        {'learning_rate' : self.args.lr})
        self.net.collect_params().zero_grad()


    def choose_action(self, s, is_ndarray=True):
        # s is channel-last, NHWC
        if not is_ndarray:
            s = np.transpose(s, (0, 3, 1, 2))
            s = nd.array(s, ctx=self.args.ctx)

        action = self.net.choose_action(s)
        action = action.asnumpy().astype(np.int32)
        return action


    def get_value(self, s, is_ndarray=True):
        # s is channel-last, NHWC
        if not is_ndarray:
            s = np.transpose(s, (0, 3, 1, 2))
            s = nd.array(s, ctx=self.args.ctx)

        value = self.net.get_value(s)
        value = value.asnumpy()
        return value


    def step(self, s):
        # s is channel-last, NHWC
        s = np.transpose(s, (0, 3, 1, 2))
        s = nd.array(s, ctx=self.args.ctx)

        value, logits = self.net(s)
        action = self.net.sample(logits)
        # logits = logits.reshape((-1, self.action_dim))
        logpac = self.net.log_prob(logits, action)

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
        obs = nd.array(obs, ctx=self.args.ctx) # .reshape((-1, self.observation_dim))
        actions = nd.array(actions, ctx=self.args.ctx).reshape((-1, 1))
        values = nd.array(values, ctx=self.args.ctx).reshape((-1, 1))
        returns = nd.array(returns, ctx=self.args.ctx).reshape((-1, 1))
        oldpi_log_prob = nd.array(logpacs, ctx=self.args.ctx).reshape((-1, 1))

        self.trainer.set_learning_rate(lrnow)

        # Auto grad
        with autograd.record():
            # Value loss
            vpred, logits = self.net(obs)
            vpred_clipped = values + nd.clip(vpred - values, -cliprange_now, cliprange_now)
            vf_loss1 = nd.square(vpred - returns)
            vf_loss2 = nd.square(vpred_clipped - returns)
            vf_loss = nd.mean(nd.maximum(vf_loss1, vf_loss2))

            # Action loss
            pi_log_prob = self.net.log_prob(logits, actions)
            ratio = nd.exp(pi_log_prob - oldpi_log_prob)
            surr1 = ratio * advantages
            surr2 = nd.clip(ratio, 1.0 - cliprange_now, 1.0 + cliprange_now) * advantages
            actor_loss = -nd.mean(nd.minimum(surr1, surr2))
            
            # Entropy term
            entropy = self.net.entropy(logits)

            # Total loss
            loss = vf_loss * self.args.value_coefficient + actor_loss \
                        - entropy * self.args.entropy_coefficient

        # Compute gradients and updates
        loss.backward()
        self.trainer.step(obs.shape[0])

        return actor_loss.asscalar(), vf_loss.asscalar(), nd.mean(entropy).asscalar()


class PPO_Gaussian(object):
    def __init__(self, env, args):
        self.action_dim = env.action_space.shape[0]
        self.observation_dim = env.observation_space.shape[0]
        self.env = env
        self.args = args

        ''' For continuous control environment (Pendulum) '''
        self.action_bound = [env.action_space.low[0], env.action_space.high[0]]
        self.net = ActorCritic_Gaussian(self.observation_dim, self.action_dim, self.action_bound, self.args)

        ''' Initialize parameters and optimizer '''
        self.net.collect_params().initialize(ctx=self.args.ctx)
        self.trainer = gluon.Trainer(self.net.collect_params(), 'adam', 
                                        {'learning_rate' : self.args.lr})
        self.net.collect_params().zero_grad()


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
        logpac = self.net.log_prob(action_nd, mu, sigma)

        logpac = logpac.asnumpy()[0]

        return value, action, logpac


    def get_value(self, s):
        s = nd.array(s, ctx=self.args.ctx)
        s = nd.reshape(s, (-1, self.observation_dim))

        value = self.net.get_value(s)
        value = value.asnumpy()[0][0]     
        return value


    # Working version with discounted returns as targets
    def old_update(self, b_s, b_a, b_r, b_logpac):
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
            pi_log_prob = self.net.log_prob(b_a, mu, sigma)
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


    # More advanced using GAE estimation as targets, not as good as discounted returns in 
    # simple environments
    def update(self, obs, returns, masks, actions, values, logpacs):
        advantages = returns - values 
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        advantages = nd.array(advantages, ctx=self.args.ctx).reshape((-1, 1))
        obs = nd.array(obs, ctx=self.args.ctx).reshape((-1, self.observation_dim))
        actions = nd.array(actions, ctx=self.args.ctx).reshape((-1, self.action_dim))
        values = nd.array(values, ctx=self.args.ctx).reshape((-1, 1))
        returns = nd.array(returns, ctx=self.args.ctx).reshape((-1, 1))
        oldpi_log_prob = nd.array(logpacs, ctx=self.args.ctx).reshape((-1, self.action_dim))

        # Learning rate scheduling
        # self.trainer.set_learning_rate(lr)

        # Auto grad
        with autograd.record():
            # Value loss
            vpred, mu, sigma = self.net(obs)
            vpred_clipped = values + nd.clip(vpred - values, -self.args.clip_param, self.args.clip_param)
            vf_loss1 = nd.square(vpred - returns)
            vf_loss2 = nd.square(vpred_clipped - returns)
            vf_loss = nd.mean(nd.maximum(vf_loss1, vf_loss2))


            # Action loss
            pi_log_prob = self.net.log_prob(actions, mu, sigma)
            ratio = nd.exp(pi_log_prob - oldpi_log_prob)
            surr1 = ratio * advantages
            surr2 = nd.clip(ratio, 1.0 - self.args.clip_param, 1.0 + self.args.clip_param) * advantages
            actor_loss = -nd.mean(nd.minimum(surr1, surr2))
            
            # Entropy term
            entropy = self.net.entropy(sigma)

            # Total loss
            loss = vf_loss * self.args.value_coefficient + actor_loss \
                        - entropy * self.args.entropy_coefficient

        # Compute gradients and updates
        loss.backward()
        self.trainer.step(obs.shape[0])
