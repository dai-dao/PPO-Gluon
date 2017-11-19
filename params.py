import mxnet as mx 


class Params(object):
    def __init__(self):
        self.ctx = mx.cpu()
        self.actor_lr = 0.001
        self.critic_lr = 0.002 
        self.clip_param = 0.2
        self.num_update_steps = 10