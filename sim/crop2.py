import numpy as np

from .simulator import Simulator


class ToyCrop2(Simulator):
    features = sorted(['precip', 'humidity', 'weight', 'radiation', 'a',
                       'precip_prev', 'humidity_prev', 'weight_prev', 'radiation_prev', 'a_prev'])

    # action is irrigation

    def __init__(self):
        super(ToyCrop2, self).__init__()

        # state feature: precip, humidity, weight, radiation
        self.state = None
        self.t = 0
        self.reset()

    def reset(self):
        precip = np.random.uniform(0, 1)
        humidity = 0.7 * precip
        radiation = np.random.uniform(0, 1)
        weight = 0
        self.t = 0
        self.state = np.array([precip, humidity, weight, radiation])
        return self.state

    def step(self, action, eps=0.1):
        assert len(action.shape) == 1
        assert action.shape[0] == 1
        irrig = action.squeeze()
        irrig = np.maximum(irrig, 0)

        precip, humidity, weight, radiation = self.state
        water = 0.6 * irrig + 0.4 * humidity
        water_gain = 1 - (water - radiation ** 2) ** 2
        water_gain = np.clip(water_gain, 0, 1)
        co2_gain = np.random.uniform(0, 1)

        new_weight = weight + 0.07 * water_gain + 0.03 * co2_gain
        new_precip = np.random.uniform(0, 1)
        new_humidity = 0.3 * humidity + 0.7 * new_precip
        new_radiation = np.random.uniform(0, 1)
        self.state = np.array([new_precip, new_humidity, new_weight, new_radiation])

        # currently, no reward
        reward = 0

        # run for 10 steps
        self.t += 1
        done = self.t >= 10

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
        features = ['precip', 'humidity', 'weight', 'radiation', 'a',
                    'precip_prev', 'humidity_prev', 'weight_prev', 'radiation_prev', 'a_prev']
        data = {f: np.array([]) for f in features}
        for _ in range(num_trajectories):
            xs, us = self.sample_trajectory()
            # batch = np.hstack((xs[:-1], us[:-1], xs[1:], us[1:]))
            for i in range(4):
                # previous step state features
                data[features[i]] = np.concatenate((data[features[i]], xs[:-1, i].flatten()))
                # current step state features
                data[features[i + 5]] = np.concatenate((data[features[i + 5]], xs[1:, i].flatten()))
            data['a_prev'] = np.concatenate((data['a_prev'], us[:-1].flatten()))
            data['a'] = np.concatenate((data['a'], us[:-1].flatten()))

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
            precip, humidity, weight, radiation = state
        else:
            assert len(state) == 3
            humidity, radiation, weight = state

        # policy: try to match optimal, but irrigate less when weight is small
        irrig = (radiation ** 2 - 0.4 * humidity) / 0.6
        irrig *= (1.6 * weight + 0.2)
        irrig = np.maximum(irrig, 0)
        return np.array([irrig])

    def get_action_prob(self, state):
        raise NotImplemented

    def seed(self, random_seed):
        np.random.seed(random_seed)
