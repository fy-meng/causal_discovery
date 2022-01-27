import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch.nn

from sim.crop import ToyCrop
from utils import Net


def get_data(num_trajectories):
    sim = ToyCrop()

    xs = None
    ys = None

    for t in range(num_trajectories):

        state_history = []
        action_history = []
        reward_history = []

        done = False
        state = sim.reset()
        while not done:
            action = sim.get_action(state)
            next_state, reward, done = sim.step(action)

            state_history.append(state)
            action_history.append(action)
            reward_history.append(reward)

            state = next_state

        state_history = np.array(state_history)
        action_history = np.array(action_history)
        reward_history = np.array(reward_history)

        x = np.hstack((state_history, action_history))
        # x = state_history

        y = np.cumsum(reward_history[::-1])[::-1]
        y = y[:, np.newaxis]

        xs = x if xs is None else np.vstack((xs, x))
        ys = y if ys is None else np.vstack((ys, y))

    return xs, ys


def train(num_trajectories=1000, num_epochs=3000, lr=1e-4):
    net = Net(6, 1, (8, 16, 32))
    xs, ys = get_data(num_trajectories)
    xs = torch.from_numpy(xs).float()
    ys = torch.from_numpy(ys).float() / 10

    criterion = torch.nn.MSELoss()
    optim = torch.optim.Adam(net.model.parameters(), lr=lr)

    losses = np.zeros(num_epochs)
    for t in range(num_epochs):
        # for data in loader:
        #     xs, ys = data[:, :6], data[:, 6:]
        optim.zero_grad()
        output = net.predict(xs)
        loss = criterion(output, ys)
        loss.backward()
        optim.step()

        losses[t] += loss.detach().cpu().numpy()

        if t % 50 == 0:
            print(f'epoch {t:04}, loss = {losses[t]}')

    net.save_model('./trained_models/crop_q_values.pth')

    plt.plot(losses)
    plt.show()


def test():
    sim = ToyCrop()
    rewards = []
    estimated = []
    done = False
    state = sim.reset()
    sim.seed(42)
    while not done:
        action = sim.get_action(state)
        next_state, reward, done = sim.step(action)

        rewards.append(reward)
        estimated.append(sim.get_action_prob(state))

        state = next_state

    rewards = np.array(rewards)
    rewards = np.cumsum(rewards[::-1])[::-1]
    plt.plot(rewards, label='truth')
    plt.plot(estimated, label='predicted')
    plt.legend()
    plt.show()

    # xs, ys = get_data(1)
    # output = net.predict(torch.from_numpy(xs).float()).detach().cpu().numpy()
    # plt.plot(ys, label='truth')
    # plt.plot(output, label='predicted')
    # plt.legend()
    # plt.show()


train()
# test()
