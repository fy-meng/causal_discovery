import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model.model import Model
from utils import verify_output_path, Net


class BiCoGAN(Model):
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
    verify_output_path('../output/legacy/generator.png')
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
    verify_output_path('../output/legacy/generator.png')
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
    verify_output_path('../output/legacy/generator.png')
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


if __name__ == '__main__':
    # test(num_epochs=500)
    # test_age_weight(300)
    # test_sine(300)
    test_net(50)
