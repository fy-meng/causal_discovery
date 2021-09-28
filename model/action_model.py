import numpy as np


class ActionModel:
    """
    Pseudo model used for action nodes in the SCM.
    """

    def __init__(self, action_func, cond_mean=None, cond_std=None, output_mean=None, output_std=None):
        """
        :param action_func: a function that maps state to action
        """
        self.action_func = action_func

        # normalization params
        self.cond_mean = cond_mean
        self.cond_std = cond_std
        self.output_mean = output_mean
        self.output_std = output_std
        assert (cond_mean is None) == (cond_std is None) == (output_mean is None) == (output_std is None), \
            'normalization parameters need to be either all None or all not None'

    def optimize(self, x, c):
        return 0.0

    def generate(self, c, z=None):
        return self.action_func(c)

    def encode(self, c, x):
        return np.zeros_like(x)

    def train(self):
        pass

    def eval(self):
        pass

    def state_dict(self):
        return {'cond_mean': self.cond_mean,
                'cond_std': self.cond_std,
                'output_mean': self.output_mean,
                'output_std': self.output_std}

    def load_state_dict(self, state_dict):
        self.cond_mean = state_dict['cond_mean']
        self.cond_std = state_dict['cond_std']
        self.output_mean = state_dict['output_mean']
        self.output_std = state_dict['output_std']

    def normalize_cond(self, c):
        if self.cond_mean is not None:
            return (c - self.cond_mean) / self.cond_std
        else:
            return c

    def normalize_output(self, x):
        if self.output_mean is not None:
            return (x - self.output_mean) / self.output_std
        else:
            return x

    def denormalize_cond(self, c):
        if self.cond_mean is not None:
            return c * self.cond_std + self.cond_mean
        else:
            return c

    def denormalize_output(self, x):
        if self.output_mean is not None:
            return x * self.output_std + self.output_mean
        else:
            return x
