import numpy as np 
import random 
import mxnet as mx 


@mx.init.register
class FaninInit(mx.init.Initializer):
    '''
        Reference implementation:
            def fanin_init(size, fanin=None):
                fanin = fanin or size[0]
                v = 1. / np.sqrt(fanin)
                return torch.Tensor(size).uniform_(-v, v)
    '''
    def __init__(self):
        super(FaninInit, self).__init__()
    def _init_weight(self, _, arr):
        fanin = arr.shape[0]
        v = 1. / np.sqrt(fanin)
        out = np.random.uniform(size=arr.shape, low=-v, high=v)
        arr[:] = out
    def _init_bias(self, _, arr):
        arr[:] = 0


def soft_update(dest, src, tau=1.0):
    dest_prefix = dest.collect_params()._prefix
    src_prefix = src.collect_params()._prefix
    
    for k, v in src.collect_params().items():
        dest_key = k.replace(src_prefix, dest_prefix)

        data = dest.collect_params()[dest_key].data() * (1.0 - tau) + \
                src.collect_params()[k].data() * tau 

        dest.collect_params()[dest_key].set_data(data)


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, action_dim, mu = 0, theta = 0.15, sigma = 0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu


    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    
    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X