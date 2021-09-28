import control
import control.matlab
import numpy as np

from sim.simulator import Simulator


class UFORotate(Simulator):
    def __init__(self, A=None, B=None, C=None, D=None, Q=None, R=None, dt=0.05):
        control.config.defaults['use_numpy_matrix'] = False

        self.features = sorted(['x_prev', 'v_prev', 'x', 'v', 'a_prev', 'a'])

        # state space properties
        A = A if A is not None else np.array([[0, 1], [0.01, 0]])
        B = B if B is not None else np.array([[0], [1]])
        C = C if C is not None else np.array([1, 0])
        D = D if D is not None else np.array([0])

        # penalize angular error and angular speed
        self.Q = Q = Q if Q is not None else np.diag([1, 1])
        # penalize thruster effort
        self.R = R = R if R is not None else np.array([1])
        # control matrix
        self.K, _, _ = control.lqr(A, B, Q, R)

        # state space model
        self.sys = control.StateSpace(A - B @ self.K, B, C, D, dt=dt)

        self.state = None
        self.t = 0
        self.reset()

    def reset(self):
        self.state = np.random.uniform(-1, 1, 2) * [np.pi, 1]
        self.t = 0

    def step(self, action, eps=1e-4):
        self.state += self.sys.dynamics(self.t, self.state, action) * self.sys.dt
        self.t += self.sys.dt
        done = np.all(np.abs(self.state) <= eps)
        reward = -self.state.T @ self.Q @ self.state - action.T * self.R * action
        return self.state, reward, done

    def sample_trajectory(self, x0=None, eps=1e-4, max_t=30) -> (np.ndarray, np.ndarray):
        # state dim is 2
        # initial state defaults to random angle in (-pi, pi), random speed in (-1, 1)
        if x0 is None:
            x = np.random.uniform(-1, 1, 2) * [np.pi, 1]
        elif not isinstance(x0, np.ndarray):
            x = np.array(x0)
        else:
            x = np.squeeze(x0)
        x = x[:, np.newaxis]
        t = 0

        xs = [x]
        us = []

        while not np.all(np.abs(x) <= eps) and t < max_t:
            u = -self.K @ x
            x = x + self.sys.dynamics(t, x, u).T * self.sys.dt
            t += self.sys.dt
            us.append(u)
            xs.append(x)

        us.append(-self.K @ x)

        return np.array(xs), np.array(us)

    def sample_batch(self, num_trajectories=100) -> np.ndarray:
        """
        :return: a batch of data sorted in name-ascending order.
        """
        data = {'x_prev': np.array([]),
                'v_prev': np.array([]),
                'x': np.array([]),
                'v': np.array([]),
                'a_prev': np.array([]),
                'a': np.array([])}
        features = ['x_prev', 'v_prev', 'x', 'v', 'a_prev', 'a']
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

    def get_action(self, x):
        return np.array(-self.K @ x).squeeze()
