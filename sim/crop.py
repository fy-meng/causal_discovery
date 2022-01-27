import numpy as np
import torch

from .simulator import Simulator
from utils import Net


class ToyCrop(Simulator):

    WATER_DECAY = 0.2

    features = sorted(['t_prev', 'rain_prev', 'sunlight_prev', 'water_prev', 'a_prev',
                       't', 'rain', 'sunlight', 'water', 'a'])

    # action is [irrigation, lighting]
    # irrigation is numerical, lighting is binary

    def __init__(self):
        super(ToyCrop, self).__init__()

        self.q_values_net = Net(6, 1, (8, 16, 32))
        self.q_values_net.model.load_state_dict(torch.load('./trained_models/crop_q_values.pth'))

        # state feature: rain, sunlight, t, water
        self.state = None
        self.sunlight_alpha = None
        self.sunlight_omega = None
        self.rains = None
        self.weight = None
        self.reset()

    def reset(self):
        # TODO: randomness?
        # np.random.seed(42)

        # parameter for sunlight
        self.sunlight_alpha = np.random.uniform(0.8, 1.5)
        self.sunlight_omega = np.random.uniform(0.7, 1)
        # self.sunlight_alpha = np.random.uniform(1.2, 1.5)
        # self.sunlight_omega = np.random.uniform(0.8, 1)

        # generate rain values
        self.rains = np.random.uniform(-0.2, 0.25, size=24)
        # self.rains = np.random.uniform(-0.10, 0.15, size=24)
        self.rains = np.cumsum(self.rains)
        self.rains = np.convolve(self.rains, np.ones(3) / 3, mode='same')
        self.rains = np.maximum(self.rains, 0)

        self.state = np.array([self.rains[0], 0, 0, 0])
        self.weight = 0

        # TODO: randomness?
        # np.random.seed(np.random.randint(65536))

        return self.state

    def step(self, action, eps=0.1):
        assert len(action.shape) == 1

        light, irrig = action
        rain, sunlight, t, water = self.state
        t = int(t)

        t += 1

        water = water * self.WATER_DECAY + (rain + irrig) * (1 - self.WATER_DECAY)
        water = np.clip(water, 0, 1)

        rad = sunlight + (0.5 if light else 0)
        rad = np.clip(rad, 0, 1)

        rain_new = self.rains[t]
        sunlight_new = self.sunlight_alpha * np.sin(t / 24 * 2 * np.pi * self.sunlight_omega)
        sunlight_new = np.maximum(sunlight_new, 0)

        self.weight += water * rad

        self.state = np.array([rain_new, sunlight_new, t, water])
        done = (t >= 23)
        reward = -0.1 * (light + irrig)
        if done:
            reward += self.weight

        return self.state, reward, done

    def sample_trajectory(self, eps=0.1, include_first_step=False) -> (np.ndarray, np.ndarray):
        self.reset()

        states = [self.state.copy()]
        actions = []

        done = False
        while not done:
            a = self.get_action(self.state)
            self.state, _, done = self.step(a, eps)

            states.append(self.state.copy())
            actions.append(a)

        actions.append(self.get_action(self.state))
        return np.array(states), np.array(actions)

    def sample_batch(self, num_trajectories=100, include_first_step=False) -> np.ndarray:
        """
        :return: a batch of data sorted in name-ascending order.
        """
        features = ['t_prev', 'rain_prev', 'sunlight_prev', 'water_prev', 'a_prev',
                    't', 'rain', 'sunlight', 'water', 'a']
        data = {f: np.array([]) for f in features}
        data['a'] = np.array([[], []]).T
        data['a_prev'] = np.array([[], []]).T
        for _ in range(num_trajectories):
            xs, us = self.sample_trajectory()
            # batch = np.hstack((xs[:-1], us[:-1], xs[1:], us[1:]))
            for i in range(0, 4):
                # previous step state features
                data[features[i]] = np.concatenate((data[features[i]], xs[:-1, i].squeeze()))
                # current step state features
                data[features[i + 5]] = np.concatenate((data[features[i + 5]], xs[1:, i].squeeze()))
            data['a_prev'] = np.concatenate((data['a_prev'], us[:-1].squeeze()))
            data['a'] = np.concatenate((data['a'], us[:-1].squeeze()))

        result = None
        for feature in sorted(features):
            if len(data[feature].shape) == 1:
                data[feature] = data[feature][:, np.newaxis]
            if result is None:
                result = data[feature]
            else:
                result = np.hstack((result, data[feature]))

        return result

    def get_action(self, state):
        if len(state) == 4:
            rain, sunlight, t, water = state
        else:  # len(state) == 2
            sunlight, water = state
        sunlight = np.maximum(sunlight, 0)
        water = np.maximum(sunlight, 0)
        light = int(sunlight < 1.0)
        irrig = np.maximum(1.0 - water, 0)
        return np.array([light, irrig])

    def get_action_prob(self, state):
        action = self.get_action(state)
        data = np.concatenate((state, action))[np.newaxis]
        output = self.q_values_net.predict(torch.from_numpy(data).float()).detach().cpu().numpy().squeeze()
        return output * 10

    def seed(self, random_seed):
        np.random.seed(random_seed)
