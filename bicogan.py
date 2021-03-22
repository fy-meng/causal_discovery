import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils import verify_output_path


# noinspection PyAbstractClass
class FCLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=True):
        super(FCLayer, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.activation = nn.ReLU(inplace=True)
        # self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.5) if dropout else None

    def forward(self, x):
        output = self.fc(x)
        output = self.activation(output)
        if self.dropout:
            output = self.dropout(output)
        return output


# noinspection PyAbstractClass
class FCNet(nn.Module):
    """
    A fully-connected neural net in which each layer is consist of Linear ->
    Activation -> Dropout, and an additional Linear layer at the end.
    """

    def __init__(self, in_channels, out_channels, hidden_layers, dropout=True, sigmoid=False):
        super(FCNet, self).__init__()
        layers = []
        prev_channels = in_channels
        for channels in hidden_layers:
            layers.append(FCLayer(prev_channels, channels, dropout=dropout))
            prev_channels = channels
        layers.append(nn.Linear(prev_channels, out_channels))
        if sigmoid:
            layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Net:
    def __init__(self, in_channels, out_channels, layers, dropout=True, sigmoid=False, lr=1e-3):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model = FCNet(in_channels, out_channels, layers, dropout=dropout, sigmoid=sigmoid)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr)

    def predict(self, *input_vars):
        input_vars = list(input_vars)
        for i in range(len(input_vars)):
            if len(input_vars[i].shape) == 1:
                input_vars[i] = input_vars[i].unsqueeze(-1)

        x = torch.cat(input_vars, dim=1)
        return self.model(x)

    def save_model(self, model_save_path):
        verify_output_path(model_save_path)
        torch.save(self.model.state_dict(), model_save_path)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()


class BiCoGAN:
    def __init__(self, cond_size, output_size, latent_size, generator_layers=(32, 64, 128),
                 encoder_layers=(32, 64, 128), discriminator_layers=(512, 256, 128, 64), sigmoid=True, **kwargs):
        self.cond_size = cond_size
        self.output_size = output_size
        self.latent_size = latent_size

        # generator maps (condition, latent) to (output)
        self.generator = Net(cond_size + latent_size, output_size, generator_layers, **kwargs)
        # encoder maps (condition, output, encoder_latent) back to (latent)
        self.encoder = Net(cond_size + output_size, latent_size, encoder_layers, **kwargs)
        # discriminator maps (condition, latent, output) to (boolean)
        self.discriminator = Net(cond_size + latent_size + output_size, 1, discriminator_layers,
                                 sigmoid=sigmoid, **kwargs)

        self.optim_iter = -1

    def optimize(self, x, c):
        loss_function = nn.BCELoss()
        batch_size = x.shape[0]

        # Data for training the discriminator
        x = x.to(self.device)
        if len(x.shape) == 1:
            x = x.unsqueeze(-1)
        c = c.to(self.device)
        if len(c.shape) == 1:
            c = c.unsqueeze(-1)

        z = torch.randn((batch_size, self.latent_size)).to(self.device)
        x_gen = self.generate(c, z)
        z_enc = self.encode(c, x)

        c_all = torch.cat((c, c))
        z_all = torch.cat((z_enc.detach(), z))
        x_all = torch.cat((x, x_gen.detach()))

        label_real = torch.ones((batch_size, 1)).to(self.device)
        label_fake = torch.zeros((batch_size, 1)).to(self.device)
        label_all = torch.cat((label_real, label_fake))

        # Training the discriminator
        self.discriminator.model.zero_grad()
        output_discriminator_all = self.discriminator.predict(c_all, z_all, x_all)
        loss_discriminator = loss_function(output_discriminator_all, label_all)
        loss_discriminator.backward()
        self.discriminator.optimizer.step()

        # Training the generator
        self.generator.model.zero_grad()
        output_discriminator_fake = self.discriminator.predict(c, z, x_gen)
        loss_generator = loss_function(output_discriminator_fake, label_real)
        loss_generator.backward()
        self.generator.optimizer.step()

        # Training the encoder
        self.encoder.model.zero_grad()
        output_discriminator_real = self.discriminator.predict(c, z_enc, x)
        loss_encoder = loss_function(output_discriminator_real, label_fake)
        loss_encoder.backward()
        self.encoder.optimizer.step()

        return loss_discriminator.detach().cpu().numpy(), \
               loss_generator.detach().cpu().numpy(), \
               loss_encoder.detach().cpu().numpy()

    def optimize_wasserstein(self, x, c):
        batch_size = x.shape[0]

        # Data for training the discriminator
        x = x.to(self.device)
        if len(x.shape) == 1:
            x = x.unsqueeze(-1)
        c = c.to(self.device)
        if len(c.shape) == 1:
            c = c.unsqueeze(-1)

        z = torch.randn((batch_size, self.latent_size)).to(self.device)
        x_gen = self.generate(c, z)
        z_enc = self.encode(c, x)

        self.optim_iter = (self.optim_iter + 1) % 5

        loss_D, loss_G, loss_E = 0, 0, 0

        # Training the discriminator
        self.discriminator.model.zero_grad()
        output_discriminator_real = self.discriminator.predict(c, z_enc.detach(), x)
        output_discriminator_fake = self.discriminator.predict(c, z, x_gen.detach())
        loss_discriminator = torch.mean(output_discriminator_fake) - torch.mean(output_discriminator_real)
        loss_discriminator.backward()
        self.discriminator.optimizer.step()
        for p in self.discriminator.model.parameters():
            p.data.clamp_(-0.01, 0.01)

        loss_D = loss_discriminator.detach().cpu().numpy()

        if self.optim_iter == 4:
            # Training the generator
            self.generator.model.zero_grad()
            output_discriminator_fake = self.discriminator.predict(c, z, x_gen)
            loss_generator = -torch.mean(output_discriminator_fake)
            loss_generator.backward()
            self.generator.optimizer.step()

            loss_G = loss_generator.detach().cpu().numpy()

            # Training the encoder
            self.encoder.model.zero_grad()
            output_discriminator_real = self.discriminator.predict(c, z_enc, x)
            loss_encoder = torch.mean(output_discriminator_real)
            loss_encoder.backward()
            self.encoder.optimizer.step()

            loss_E = loss_encoder.detach().cpu().numpy()

        return loss_D, loss_G, loss_E

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

        return self.generator.predict(c, z)

    def encode(self, c, x):
        if not isinstance(c, torch.Tensor):
            c = torch.Tensor(c).to(self.device)
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x).to(self.device)
        return self.encoder.predict(c, x)

    def train(self):
        self.generator.train()
        self.encoder.train()
        self.discriminator.train()

    def eval(self):
        self.generator.eval()
        self.encoder.eval()
        self.discriminator.eval()

    def state_dict(self):
        return {'discriminator': self.discriminator.model.state_dict(),
                'generator': self.generator.model.state_dict(),
                'encoder': self.encoder.model.state_dict()}

    def load_state_dict(self, state_dict):
        assert len(state_dict) == 3 \
               and tuple(state_dict.keys()) == ('discriminator', 'generator', 'encoder')
        self.discriminator.model.load_state_dict(state_dict['discriminator'])
        self.generator.model.load_state_dict(state_dict['generator'])
        self.encoder.model.load_state_dict(state_dict['encoder'])


class LinearNoiseGenerator:
    def __init__(self, cond_size, output_size, latent_size, layers=(16, 32), **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cond_size = cond_size
        self.output_size = output_size
        self.latent_size = latent_size

        if self.cond_size > 0:
            self.net = Net(in_channels=cond_size, out_channels=output_size, layers=layers, **kwargs)
        else:
            self.net = None

        # assumes additive noise, needs to record variance
        self.num_samples = 0
        self.mean = 0
        self.var = 0

    def optimize(self, x, c):
        batch_size = x.shape[0]

        # update mean and variance
        x_ = x.detach().cpu().numpy()
        n, m = self.num_samples, batch_size
        mu_n, mu_m = self.mean, np.mean(x_)
        var_n, var_m = self.var, np.var(x_)
        self.var = (n * var_n + m * var_m) / (n + m) + (n * m) * (((mu_n - mu_m) / (n + m)) ** 2)
        self.mean = (n * mu_n + m * mu_m) / (n + m)
        self.num_samples += batch_size

        # only train the model if not the source
        if self.cond_size > 0:
            loss_function = nn.MSELoss()

            x = x.to(self.device)
            if len(x.shape) == 1:
                x = x.unsqueeze(-1)
            c = c.to(self.device)
            if len(c.shape) == 1:
                c = c.unsqueeze(-1)

            z = torch.randn((batch_size, self.latent_size)).to(self.device)

            self.net.model.zero_grad()
            output = self.net.predict(c)
            loss = loss_function(x, output)
            loss.backward()
            self.net.optimizer.step()

            return loss

        else:
            return 0

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

        if self.cond_size > 0:
            return self.net.predict(c) + np.sqrt(self.var) * z
        else:
            return np.sqrt(self.var) * z

    def encode(self, c, x):
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

    def train(self):
        if self.net:
            self.net.train()

    def eval(self):
        if self.net:
            self.net.eval()

    def state_dict(self):
        if self.cond_size > 0:
            return {'state_dict': self.net.model.state_dict(),
                    'mean': self.mean,
                    'var': self.var}
        else:
            return {'mean': self.mean,
                    'var': self.var}

    def load_state_dict(self, state_dict):
        if self.cond_size > 0:
            self.net.model.load_state_dict(state_dict['state_dict'])
        self.mean = state_dict['mean']
        self.var = state_dict['var']


class LinearNoiseRegressor:
    def __init__(self, cond_size, output_size, latent_size, **kwargs):
        self.cond_size = cond_size
        self.output_size = output_size
        self.latent_size = latent_size

        self.w = np.zeros((cond_size, output_size))

        # assumes additive noise, needs to record variance
        self.num_samples = 0
        self.mean = 0
        self.var = 0

    def optimize(self, x, c):
        # update mean and variance
        self.var = np.var(x)
        self.mean = np.mean(x)
        self.num_samples += x.shape[0]

        if self.cond_size > 0:
            self.w = np.linalg.inv(c.T @ c) @ c.T @ x

        # placeholder
        loss = np.linalg.norm(x - c @ self.w)
        return loss

    def generate(self, c, z=None):
        if len(c.shape) == 1:
            c = c[np.newaxis]
        batch_size = c.shape[0]
        if z is None:
            z = np.random.randn(batch_size)

        # if self.cond_size > 0:
        #     print(c.shape, self.w.shape)
        #     return c @ self.w + np.sqrt(self.var) * z
        # else:
        #     return np.sqrt(self.var) * z

        if self.cond_size > 0:
            return c @ self.w
        else:
            return np.zeros(batch_size)

    def encode(self, c, x):
        if len(c.shape) == 1:
            c = c[np.newaxis]
        if len(x.shape) == 1:
            x = x[np.newaxis]

        if self.cond_size > 0:
            return (x - c @ self.w) / np.sqrt(self.var)
        else:
            return x / np.sqrt(self.var)

    def state_dict(self):
        if self.cond_size > 0:
            return {'w': self.w,
                    'mean': self.mean,
                    'var': self.var}
        else:
            return {'mean': self.mean,
                    'var': self.var}

    def load_state_dict(self, state_dict):
        if self.cond_size > 0:
            self.w = state_dict['w']
        self.mean = state_dict['mean']
        self.var = state_dict['var']


def test(num_epochs):
    # x ~ 3 * N(0, 1)
    # y = -4 * x + 1 * N(0, 1)
    gan = BiCoGAN(1, 1, 1)
    loss_gs = []
    loss_ds = []
    loss_es = []
    for epoch in range(num_epochs):
        train_data_length = 1024
        train_data = np.zeros((train_data_length, 2, 1))
        train_data[:, 0] = 3 * np.random.randn(train_data_length, 1)
        train_data[:, 1] = -4 * train_data[:, 0] + np.random.randn(train_data_length, 1)
        train_data = torch.tensor(train_data).float()
        train_labels = torch.zeros(train_data_length)
        train_set = [
            (train_data[i], train_labels[i]) for i in range(train_data_length)
        ]

        batch_size = 32
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        for n, (samples, _) in enumerate(train_loader):
            samples = samples
            c = samples[:, 0]
            x = samples[:, 1]

            loss_D, loss_G, loss_E = gan.optimize(x, c)

            loss_ds.append(loss_D)
            loss_gs.append(loss_G)
            loss_es.append(loss_E)

        if epoch % 10 == 0:
            print(f'epoch {epoch:03d}: loss_D = {loss_D:.4f}, loss_G = {loss_G:.4f}, loss_E = {loss_E:.4f}')

    # testing

    import matplotlib.pyplot as plt

    # plot loss
    plt.plot(np.arange(501, len(loss_ds) + 1), loss_ds[500:], label='discriminator loss')
    plt.plot(np.arange(501, len(loss_ds) + 1), loss_gs[500:], label='generator loss')
    plt.plot(np.arange(501, len(loss_ds) + 1), loss_es[500:], label='encoder loss')
    plt.legend()
    verify_output_path('./output/generator.png')
    plt.savefig('./output/loss.png')
    plt.clf()

    # plot truth vs generated
    c = 3 * np.random.randn(256, 1)
    x = -4 * c + np.random.randn(256, 1)

    print('groud truth range:')
    print(np.min(x), np.max(x))

    plt.scatter(c, x, label='ground truth')

    gan.eval()
    z = np.random.randn(256, 1)
    x_gen = gan.generate(c, z).detach().cpu().numpy()

    print('generated output range:')
    print(np.min(x_gen), np.max(x_gen))

    plt.scatter(c, x_gen, label='generator')
    plt.legend()
    plt.xlabel('condition')
    plt.ylabel('output')
    plt.savefig('./output/generator.png')
    plt.clf()

    # plot generated vs encoded
    fig = plt.figure()
    ax: plt.Axes = plt.axes(projection='3d')
    ax.scatter(c, z, x_gen, label='real condition/latent + generated output')

    z_enc = gan.encode(c, x_gen)
    z_enc = z_enc.detach().cpu().numpy()

    ax.scatter(c, z_enc, x, label='encoded condition/latent + real output')
    ax.legend()
    ax.set_xlabel('condition')
    ax.set_ylabel('latent')
    ax.set_zlabel('output')
    plt.savefig('./output/encoder.png')
    plt.clf()

    # testing the difference between generator and encoder
    print(f'average latent encoding error:')
    print(np.sum(np.abs(z_enc - z)) / len(z_enc))
    x_enc_gen = gan.generate(c, z_enc).detach().cpu().numpy()
    print(f'average distance between x_gen and x_enc_gen:')
    print(np.sum(np.abs(x_gen - x_enc_gen)) / len(x_gen))

    plt.scatter(z, x_gen, label='gen')
    plt.scatter(z_enc, x, label='enc')
    plt.legend()
    plt.show()


def test_age_weight(num_epochs):
    gan = BiCoGAN(1, 1, 8, sigmoid=True)

    train_data_length = 1024
    train_data = np.zeros((train_data_length, 2, 1))
    # age = np.random.uniform(20, 80, size=train_data_length)
    # weight = np.where(age <= 40,
    #                   0.25 * age + 60,
    #                   -0.25 * age + 80
    #                   ) + np.random.uniform(-2, 2, size=train_data_length)
    age = 30 * np.random.randn(train_data_length) + 50
    age = np.clip(age, 20, 80)
    weight = np.where(age <= 40,
                      0.25 * age + 60,
                      -0.25 * age + 80
                      ) + 6 * np.random.rand(train_data_length)
    weight = np.clip(weight, 40, 80)

    train_data[:, 0] = age[:, None]
    train_data[:, 1] = weight[:, None]
    train_data = torch.tensor(train_data).float()
    train_labels = torch.zeros(train_data_length)
    train_set = [
        (train_data[i], train_labels[i]) for i in range(train_data_length)
    ]

    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    loss_gs = []
    loss_ds = []
    loss_es = []
    for epoch in range(num_epochs):

        for n, (samples, _) in enumerate(train_loader):
            samples = samples
            c = samples[:, 0]
            x = samples[:, 1]

            loss_D, loss_G, loss_E = gan.optimize(x, c)

            loss_ds.append(loss_D)
            loss_gs.append(loss_G)
            loss_es.append(loss_E)

        if epoch % 10 == 0:
            print(f'epoch {epoch:03d}: loss_D = {loss_D:.4f}, loss_G = {loss_G:.4f}, loss_E = {loss_E:.4f}')

    # testing
    gan.eval()

    import matplotlib.pyplot as plt

    # plot loss
    plt.plot(np.arange(501, len(loss_ds) + 1), loss_ds[500:], label='discriminator loss')
    plt.plot(np.arange(501, len(loss_ds) + 1), loss_gs[500:], label='generator loss')
    plt.plot(np.arange(501, len(loss_ds) + 1), loss_es[500:], label='encoder loss')
    plt.legend()
    verify_output_path('./output/generator.png')
    plt.savefig('./output/loss.png')
    plt.clf()

    # plot truth vs generated
    c = 30 * np.random.randn(1024) + 50
    c = np.clip(c, 20, 80)
    x = np.where(c <= 40,
                 0.25 * c + 60,
                 -0.25 * c + 80
                 ) + 6 * np.random.rand(1024)
    x = np.clip(x, 40, 80)

    plt.scatter(c, x, label='ground truth')

    gan.eval()
    z = np.random.randn(1024, 8)
    x_gen = gan.generate(c[:, None], z).detach().cpu().numpy()

    print('generated output range:')
    print(np.min(x_gen), np.max(x_gen))

    plt.scatter(c, x_gen, label='generator')
    plt.legend()
    plt.xlabel('condition')
    plt.ylabel('output')
    plt.savefig('./output/generator.png')
    plt.clf()


def test_sine(num_epochs):
    gan = BiCoGAN(1, 1, 4)

    train_data_length = 1024
    train_data = np.zeros((train_data_length, 2, 1))
    train_data[:, 0] = 3 * np.random.randn(train_data_length)[:, None]
    train_data[:, 1] = np.sin(-train_data[:, 0] + 0.5) + 0.5 * np.random.randn(train_data_length)[:, None]
    train_data = torch.tensor(train_data).float()
    train_labels = torch.zeros(train_data_length)
    train_set = [
        (train_data[i], train_labels[i]) for i in range(train_data_length)
    ]

    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    loss_gs = []
    loss_ds = []
    loss_es = []
    for epoch in range(num_epochs):

        for n, (samples, _) in enumerate(train_loader):
            samples = samples
            c = samples[:, 0]
            x = samples[:, 1]

            loss_D, loss_G, loss_E = gan.optimize(x, c)

            loss_ds.append(loss_D)
            loss_gs.append(loss_G)
            loss_es.append(loss_E)

        if epoch % 10 == 0:
            print(f'epoch {epoch:03d}: loss_D = {loss_D:.4f}, loss_G = {loss_G:.4f}, loss_E = {loss_E:.4f}')

    # testing

    import matplotlib.pyplot as plt

    # plot loss
    plt.plot(np.arange(501, len(loss_ds) + 1), loss_ds[500:], label='discriminator loss')
    plt.plot(np.arange(501, len(loss_ds) + 1), loss_gs[500:], label='generator loss')
    plt.plot(np.arange(501, len(loss_ds) + 1), loss_es[500:], label='encoder loss')
    plt.legend()
    verify_output_path('./output/generator.png')
    plt.savefig('./output/loss.png')
    plt.clf()

    # plot truth vs generated
    c = 3 * np.random.randn(train_data_length)
    x = np.sin(c + 0.5) + 0.5 * np.random.randn(train_data_length)

    plt.scatter(c, x, label='ground truth')

    gan.eval()
    z = np.random.randn(1024, 4)
    x_gen = gan.generate(c[:, None], z).detach().cpu().numpy()

    print('generated output range:')
    print(np.min(x_gen), np.max(x_gen))

    plt.scatter(c, x_gen, label='generator')
    plt.legend()
    plt.xlabel('condition')
    plt.ylabel('output')
    plt.savefig('./output/generator.png')
    plt.clf()


def test_net(num_epochs):
    def generate_data(num_trajectories):
        # features: x (location), s (measured distance to obstacle), v (velocity)
        # action: a (acceleration)
        # action only consider s
        data = []

        for _ in range(num_trajectories):
            d = np.random.uniform(20, 80)
            a_max = 0.5
            v_max = 2

            x = 0
            s = d - x + np.random.randn()
            v = 0
            a = a_max if s >= (v_max ** 2) / (2 * a_max) else -a_max
            t = np.random.uniform(0.5, 1)

            prev_state = [a, s, t, v, x]

            while x < d:
                x += v * t + 0.5 * a * (t ** 2)
                v += a * t
                s = d - x + np.random.randn()
                a = a_max if s >= (v_max ** 2) / (2 * a_max) else -a_max

                state = [a, s, t, v, x]
                data.append(prev_state + state)
                prev_state = state

        return np.array(data)
    data = generate_data(500)
    data = torch.tensor(data).float()
    x = data[:, 6]
    y = data[:, 5]
    train_set = [(x[i], y[i]) for i in range(x.shape[0])]

    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # train
    net = LinearNoiseGenerator(1, 1, 1, layers=(16,))

    losses = []
    for epoch in range(num_epochs):
        loss = 0
        for n, (x, y) in enumerate(train_loader):
            loss = net.optimize(y, x)
            losses.append(loss)

        if epoch % 10 == 0:
            print(f'epoch {epoch:03d}: loss = {loss:.4f}')

    import matplotlib.pyplot as plt
    x = data[:, 6].detach().cpu().numpy()
    y = data[:, 5].detach().cpu().numpy()
    print(x.shape, y.shape)
    plt.scatter(x, y, label='ground truth', c='orange')
    net.eval()
    x_ = np.linspace(0, 80, 300)[:, np.newaxis]
    y_ = net.generate(c=x_)
    plt.plot(x_, y_.detach().cpu().numpy(), label='estimated')
    plt.legend()
    plt.plot()
    plt.show()


if __name__ == '__main__':
    # test(num_epochs=500)
    # test_age_weight(300)
    # test_sine(300)
    test_net(50)
