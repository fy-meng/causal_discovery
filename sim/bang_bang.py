import numpy as np

from .simulator import Simulator


class BangBangControl(Simulator):
    features = sorted(['v_prev', 'x_prev', 's_prev', 'v', 'x', 's', 'a_prev', 'a'])

    def __init__(self, a_max=0.5, v_max=2, dt=0.175, noise=False):
        self.a_max = a_max
        self.v_max = v_max
        self.dt = dt
        self.noise = noise

        # state feature: v, x, s
        self.state = None
        self.d = None
        self.reset()

    def reset(self):
        self.d = np.clip(3 * np.random.randn() + 30, 20, None)
        # s = self.d + np.random.randn()
        s = self.d
        self.state = np.array([0, 0, s])

    def step(self, action, eps=0.1):
        assert len(action.shape) == 1
        a = action[0]

        v, x, s = self.state

        x += v * self.dt + 0.5 * a * (self.dt ** 2)

        v += a * self.dt
        v = min(v, self.v_max)

        s = self.d - x
        if self.noise:
            s += 2 * np.random.randn()

        self.state = np.array([v, x, s])
        done = np.abs(x - self.d) <= eps
        reward = -1 if not done else 5000

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
        data = {'v_prev': np.array([]),
                'x_prev': np.array([]),
                's_prev': np.array([]),
                'v': np.array([]),
                'x': np.array([]),
                's': np.array([]),
                'a_prev': np.array([]),
                'a': np.array([])}
        features = ['v_prev', 'x_prev', 's_prev', 'v', 'x', 's', 'a_prev', 'a']
        for _ in range(num_trajectories):
            xs, us = self.sample_trajectory()
            batch = np.hstack((xs[:-1], xs[1:], us[:-1], us[1:]))
            for i, feature in enumerate(features):
                data[feature] = np.concatenate((data[feature], batch[:, i].squeeze()))

        result = None
        for feature in sorted(features):
            if result is None:
                result = data[feature][:, np.newaxis]
            else:
                result = np.hstack((result, data[feature][:, np.newaxis]))

        return result

    def get_action(self, state):
        if len(state) == 3:
            v, x, s = state
        else:
            s = state[0]
        a = self.a_max if s >= (self.v_max ** 2) / (2 * self.a_max) else - self.a_max
        return np.array([a])

    def get_action_prob(self, state):
        return self.get_action(state)

    def seed(self, random_seed):
        np.random.seed(random_seed)
