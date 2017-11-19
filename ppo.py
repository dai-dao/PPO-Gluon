import numpy as np 
import mxnet as mx 
from mxnet import gluon, autograd, nd
from model import Actor, Critic
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

        self.critic = Critic(self.observation_dim, self.action_dim, self.action_bound, self.args)
        self.actor = Actor(self.observation_dim, self.action_dim, self.action_bound, self.args)
        self.old_actor = Actor(self.observation_dim, self.action_dim, self.action_bound, self.args)


        self.critic.collect_params().initialize(mx.init.Xavier(magnitude=1.24), ctx=self.args.ctx)
        self.actor.collect_params().initialize(mx.init.Xavier(magnitude=1.24), ctx=self.args.ctx)
        self.old_actor.collect_params().initialize(mx.init.Xavier(magnitude=1.24), ctx=self.args.ctx)


        # Copy params from new to old
        soft_update(self.old_actor, self.actor)

        self.actor_trainer = gluon.Trainer(self.actor.collect_params(), 'adam', 
                                        {'learning_rate' : self.args.actor_lr})
        self.critic_trainer = gluon.Trainer(self.critic.collect_params(), 'adam',
                                        {'learning_rate' : self.args.critic_lr})
        

    def sample(self, mu, sigma):
        epsilon = nd.random_normal(shape=mu.shape, loc=0., scale=1., ctx=self.args.ctx)
        out = mu + sigma * epsilon
        return out

    
    def log_gaussian(self, x, mu, sigma):   
        out = -0.5 * np.log(2.0 * np.pi) - nd.log(sigma + 1e-5) - (x - mu) ** 2 / (2 * sigma ** 2 + 1e-5)
        return out


    def choose_action(self, s):
        s = nd.array(s, ctx=self.args.ctx)
        s = nd.reshape(s, (-1, self.observation_dim))

        mu, sigma = self.actor(s)
        action = self.sample(mu, sigma)
        action = action.asnumpy()[0]
        # Add noise to aid exploration
        # The noise IS the problem
        #action = action + (self.noise.sample() * 2) # Scaled by max possible action
        # Leave clip on for now 
        action = np.clip(action, self.action_bound[0], self.action_bound[1])
        return action


    def get_value(self, s):
        s = nd.array(s, ctx=self.args.ctx)
        s = nd.reshape(s, (-1, self.observation_dim))

        value = self.critic(s)
        value = value.asnumpy()[0]     
        return value


    def update(self, b_s, b_a, b_r):
        # Copy params from new to old
        soft_update(self.old_actor, self.actor)

        b_s = nd.array(b_s, ctx=self.args.ctx).reshape((-1, self.observation_dim))
        b_a = nd.array(b_a, ctx=self.args.ctx).reshape((-1, self.action_dim))
        b_r = nd.array(b_r, ctx=self.args.ctx).reshape((-1, 1))

        old_mu, old_sigma = self.old_actor(b_s)
        oldpi_log_prob = self.log_gaussian(b_a, old_mu, old_sigma)
        
        for _ in range(self.args.num_update_steps):
            with autograd.record():
                # Value loss
                v_pred = self.critic(b_s)
                advantage = b_r - v_pred
                vf_loss = nd.mean(nd.square(advantage))

            vf_loss.backward()
            self.critic_trainer.step(b_s.shape[0])

            with autograd.record():
                # Action loss
                mu, sigma = self.actor(b_s)
                pi_log_prob = self.log_gaussian(b_a, mu, sigma)
                ratio = nd.exp(pi_log_prob - oldpi_log_prob)
                surr1 = ratio * advantage
                surr2 = nd.clip(ratio, 1.0 - self.args.clip_param, 1.0 + self.args.clip_param) * advantage
                actor_loss = -nd.mean(nd.minimum(surr1, surr2))

            actor_loss.backward()
            self.actor_trainer.step(b_s.shape[0])