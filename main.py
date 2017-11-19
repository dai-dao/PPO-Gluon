import gym 
from ppo import PPO
from params import Params
import numpy as np 
import mxnet as mx 

env = gym.make('Pendulum-v0').unwrapped
params = Params()
ppo = PPO(env, params)
all_ep_r = []


EP_MAX = 200
EP_LEN = 200
GAMMA = 0.9
LR = 0.0001
BATCH = 32
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10


# Seed
env.seed(1)
np.random.seed(1)
mx.random.seed(1)


for ep in range(EP_MAX):
    s = env.reset()
    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0
    for t in range(EP_LEN):    # in one episode
        env.render()
        a = ppo.choose_action(s)
        s_, r, done, _ = env.step(a)
        buffer_s.append(s)
        buffer_a.append(a)
        buffer_r.append((r+8)/8)    # normalize reward, find to be useful
        s = s_
        ep_r += r

        # update ppo
        if (t+1) % BATCH == 0 or t == EP_LEN-1:
            v_s_ = ppo.get_value(s_)
            discounted_r = []
            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()

            bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
            buffer_s, buffer_a, buffer_r = [], [], []

            ppo.update(bs, ba, br)
            
    if ep == 0: 
        all_ep_r.append(ep_r)
    else: 
        all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)

    print('Ep: %i' % ep, "|Ep_r: %i" % ep_r)    