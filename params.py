import mxnet as mx 


class Breakout_Params(object):
    def __init__(self):
        self.ctx = mx.cpu()
        self.log_interval = 1

        self.lr = 2.5e-4
        self.value_coefficient = 0.5
        self.entropy_coefficient = 0.01
        self.gamma = 0.99 
        self.lam = 0.95

        self.nsteps = 128
        self.nminibatches = 4
        self.num_update_epochs = 4
        self.lr_schedule = lambda x : x * 2.5e-4
        self.clip_range_schedule = lambda x : x * 0.1
        self.num_timesteps = int(10e6 * 1.1)


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
                    

        self.nsteps = 64 # Number of steps in one roll-out
        self.num_update_steps = 10 # 
        self.num_update_epochs = 10 # Number of parameter updates to do on 1 batch
        self.nenvs = 1 # Number of parallel environments to run
        self.num_timesteps = 10e6 # Total number of steps to take 
        self.nminibatches = 2 # Number of batches in one roll-out to train
        self.clip_range = lambda f : f * 0.1
        self.lr_schedule = lambda f : f * 2.5e-4


class CartPole_Params(object):
    def __init__(self):
        self.ctx = mx.cpu()
        self.actor_lr = 0.001
        self.critic_lr = 0.002 
        self.clip_param = 0.2
        self.value_coefficient = 0.5
        self.entropy_coefficient = 0.01
        self.gamma = 0.99 # Reward discount [0.9, 0.99]
        self.lam = 0.95
                    

        self.nsteps = 256 # Number of steps in one roll-out
        self.num_update_steps = 10 # 
        self.num_update_epochs = 10 # Number of parameter updates to do on 1 batch
        self.nenvs = 1 # Number of parallel environments to run
        self.num_timesteps = 10e6 # Total number of steps to take 
        self.nminibatches = 4 # Number of batches in one roll-out to train
        self.clip_range = lambda f : f * 0.1
        self.lr_schedule = lambda f : f * 2.5e-4