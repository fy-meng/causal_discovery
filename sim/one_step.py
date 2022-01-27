import numpy as np

from .simulator import Simulator


class BangBangControl(Simulator):
    features = sorted(['s1', 's2', 's3', 'vp', 'a'])

    def __init__(self, c1=1, c2=2, c3=3, c12=4):
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c12 = c12

    def reset(self):
        pass

    def step(self, action, eps=0.1):
        return None, None, True

    def sample_trajectory(self, eps=0.1, include_first_step=False) -> (np.ndarray, np.ndarray):
        return None

    def sample_batch(self, num_trajectories=100, include_first_step=False) -> np.ndarray:
        return None

    def get_action(self, state):
        assert len()
        return self.c1

    def get_action_prob(self, state):
        return self.get_action(state)

    def seed(self, random_seed):
        np.random.seed(random_seed)



