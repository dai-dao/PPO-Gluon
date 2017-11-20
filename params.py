import mxnet as mx 


class Pendulum_Params(object):
    def __init__(self):
        self.ctx = mx.cpu()
        self.actor_lr = 0.001
        self.critic_lr = 0.002 
        self.clip_param = 0.2
        self.value_coefficient = 0.5
        self.entropy_coefficient = 0.01
        self.gamma = 0.9 # Reward discount [0.9, 0.99]
        self.lam = 0.95
                    

        self.nsteps = 32 # Number of steps in one roll-out
        self.num_update_steps = 10 # 
        self.num_update_epochs = 10 # Number of parameter updates to do on 1 batch
        self.nenvs = 1 # Number of parallel environments to run
        self.num_timesteps = 10e6 # Total number of steps to take 
        self.nminibatches = 1 # Number of batches in one roll-out to train
        self.clip_range = lambda f : f * 0.1
        self.lr_schedule = lambda f : f * 2.5e-4