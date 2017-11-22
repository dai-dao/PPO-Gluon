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


@mx.init.register
class OrthoInit(mx.init.Initializer):
    def __init__(self, scale=1.0):
        super(OrthoInit, self).__init__()
        self.scale = scale

    def _init_weight(self, _, arr):
        shape = arr.shape
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: 
            # assumes NHWC
            # flat_shape = (np.prod(shape[:-1]), shape[-1])
            # assumes NCHW
            flat_shape = (np.prod([shape[0], shape[2], shape[3]]), shape[1])

        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        out = (self.scale * q[:shape[0], :shape[1]]).astype(np.float32)
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


def flatten_env_vec(arr):
    '''
        arr shape is [num_env, num_steps, ob_shape]
        is the swap necessary??
    '''
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])