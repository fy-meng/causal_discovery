import numpy as np

from model.model import Model


class LinearRegressor(Model):
    def __init__(self, cond_size, output_size, latent_size,
                 cond_mean=None, cond_std=None, output_mean=None, output_std=None, **kwargs):
        self.cond_size = cond_size
        self.output_size = output_size
        self.latent_size = latent_size

        self.w = None

        self.cond_mean = cond_mean
        self.cond_std = cond_std
        self.output_mean = output_mean
        self.output_std = output_std
        assert (cond_mean is None) == (cond_std is None) == (output_mean is None) == (output_std is None), \
            'normalization parameters need to be either all None or all not None'

        # assumes additive noise, needs to record variance
        self.num_samples = 0
        self.var = 0

    def optimize(self, x, c):
        c = self.normalize_cond(c)
        x = self.normalize_output(x)

        if len(x.shape) == 1:
            x = x[:, np.newaxis]

        batch_size = c.shape[0]
        c = np.hstack((c, np.ones((batch_size, 1))))

        self.w = np.linalg.inv(c.T @ c) @ c.T @ x

        error = x - c @ self.w
        self.var = np.var(error)

        # TODO: is this correct? how to handle var = 0
        if np.isclose(self.var, 0):
            self.var = 1

        loss = np.mean(error ** 2)
        return loss

    def generate(self, c, z=None):
        # normalization
        c = self.normalize_cond(c)

        if len(c.shape) == 1:
            c = c[np.newaxis, :]

        batch_size = c.shape[0]

        if z is None:
            z = np.randn(batch_size, self.latent_size)
        if len(z.shape) == 1:
            z = z[:, np.newaxis]

        if self.cond_size > 0:
            c = np.hstack((c, np.ones((batch_size, 1))))
            x = c @ self.w + np.sqrt(self.var) * z
        else:
            x = np.sqrt(self.var) * z

        x = x.squeeze()
        # de-normalization
        x = self.denormalize_output(x)
        return x

    def encode(self, c, x):
        c = self.normalize_cond(c)
        x = self.normalize_output(x)

        if len(c.shape) == 1:
            c = c[np.newaxis, :]
        if len(x.shape) == 1:
            x = x[:, np.newaxis]

        if self.cond_size > 0:
            batch_size = c.shape[0]
            c = np.hstack((c, np.ones((batch_size, 1))))
            return (x - c @ self.w).squeeze() / np.sqrt(self.var)
        else:
            return x / np.sqrt(self.var)

    def normalize_cond(self, c):
        if self.cond_mean is not None:
            if np.all(np.isclose(self.cond_std, 0)):
                return c - self.output_mean
            else:
                return (c - self.cond_mean) / self.cond_std
        else:
            return c

    def normalize_output(self, x):
        if self.output_mean is not None:
            if np.all(np.isclose(self.output_std, 0)):
                return x - self.output_mean
            else:
                return (x - self.output_mean) / self.output_std
        else:
            return x

    def denormalize_cond(self, c):
        if self.cond_mean is not None:
            if np.all(np.isclose(self.cond_std, 0)):
                return c + self.cond_mean
            else:
                return c * self.cond_std + self.cond_mean
        else:
            return c

    def denormalize_output(self, x):
        if self.output_mean is not None:
            if np.all(np.isclose(self.output_std, 0)):
                return x + self.output_mean
            else:
                return x * self.output_std + self.output_mean
        else:
            return x

    def train(self):
        pass

    def eval(self):
        pass

    def state_dict(self):
        return {'w': self.w,
                'cond_mean': self.cond_mean,
                'cond_std': self.cond_std,
                'output_mean': self.output_mean,
                'output_std': self.output_std,
                'var': self.var}

    def load_state_dict(self, state_dict):
        self.w = state_dict['w']
        self.cond_mean = state_dict['cond_mean']
        self.cond_std = state_dict['cond_std']
        self.output_mean = state_dict['output_mean']
        self.output_std = state_dict['output_std']
        self.var = state_dict['var']
