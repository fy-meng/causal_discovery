import numpy as np
import torch
import torch.nn as nn

from model.model import Model
from utils import Net


class LinearNoiseGenerator(Model):
    def __init__(self, cond_size, output_size, latent_size, layers=(4, 4, 4),
                 cond_mean=None, cond_std=None, output_mean=None, output_std=None, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cond_size = cond_size
        self.output_size = output_size
        self.latent_size = latent_size

        # normalization params
        self.cond_mean = cond_mean
        self.cond_std = cond_std
        self.output_mean = output_mean
        self.output_std = output_std
        assert (cond_mean is None) == (cond_std is None) == (output_mean is None) == (output_std is None), \
            'normalization parameters need to be either all None or all not None'

        if self.cond_size > 0:
            self.net = Net(in_channels=cond_size, out_channels=output_size, layers=layers, **kwargs)
        else:
            self.net = None

        # assumes additive noise, needs to record variance
        self.num_samples = 0
        self.var = 0

    def optimize(self, x, c):
        batch_size = x.shape[0]

        # apply normalization
        x = self.normalize_output(x)
        c = self.normalize_cond(c)

        # convert to tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x).to(self.device)
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c).to(self.device)

        # force data types
        x = x.float()
        c = c.float()

        # only train the model if not the source
        loss = 0.0
        if self.cond_size > 0:
            loss_function = nn.MSELoss()

            x = x.to(self.device)
            if len(x.shape) == 1:
                x = x.unsqueeze(-1)
            c = c.to(self.device)
            if len(c.shape) == 1:
                c = c.unsqueeze(-1)

            self.net.model.zero_grad()
            output = self.net.predict(c)
            loss = loss_function(x, output)
            loss.backward()
            self.net.optimizer.step()

            loss = loss.detach().cpu().numpy()
            error = output - x
        else:
            error = x

        # update variance using prediction error
        error = error.detach().cpu().numpy()
        self.var = (0.2 * self.num_samples * self.var + 0.8 * batch_size * np.var(error)) \
                   / (0.2 * self.num_samples + 0.8 * batch_size)
        self.num_samples += batch_size

        return loss

    def generate(self, c, z=None):
        if not isinstance(c, torch.Tensor):
            c = torch.Tensor(c).to(self.device)
        if len(c.shape) == 1:
            c = c.unsqueeze(0)
        batch_size = c.shape[0]
        if z is None:
            z = torch.randn(batch_size, self.latent_size).to(self.device)
        elif not isinstance(z, torch.Tensor):
            z = torch.Tensor(z).to(self.device)

        # normalization
        c = self.normalize_cond(c)

        # force data type
        c = c.float()

        if self.cond_size > 0:
            x = self.net.predict(c) + np.sqrt(self.var) * z
        else:
            x = np.sqrt(self.var) * z

        # de-normalization
        x = self.denormalize_output(x)
        return x

    def encode(self, c, x):
        # normalization
        c = self.normalize_cond(c)
        x = self.normalize_output(x)

        if not isinstance(c, torch.Tensor):
            c = torch.Tensor(c).to(self.device)
        if len(c.shape) == 1:
            c = c.unsqueeze(0)
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x).to(self.device)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        if self.cond_size > 0:
            return (x - self.net.predict(c)) / np.sqrt(self.var)
        else:
            return x / np.sqrt(self.var)

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

    def train(self):
        if self.net:
            self.net.train()

    def eval(self):
        if self.net:
            self.net.eval()

        # TODO: temp fix for zero var
        if self.var == 0:
            self.var = 1

    def state_dict(self):
        if self.cond_size > 0:
            return {'state_dict': self.net.model.state_dict(),
                    'cond_mean': self.cond_mean,
                    'cond_std': self.cond_std,
                    'output_mean': self.output_mean,
                    'output_std': self.output_std,
                    'var': self.var}
        else:
            return {'cond_mean': self.cond_mean,
                    'cond_std': self.cond_std,
                    'output_mean': self.output_mean,
                    'output_std': self.output_std,
                    'var': self.var}

    def load_state_dict(self, state_dict):
        if self.cond_size > 0:
            self.net.model.load_state_dict(state_dict['state_dict'])
        self.cond_mean = state_dict['cond_mean']
        self.cond_std = state_dict['cond_std']
        self.output_mean = state_dict['output_mean']
        self.output_std = state_dict['output_std']
        self.var = state_dict['var']
