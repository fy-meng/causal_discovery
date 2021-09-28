class Simulator:
    features = None

    def reset(self):
        raise NotImplemented

    def step(self, action, eps=None):
        raise NotImplemented

    def get_action(self, state):
        raise NotImplemented

    def get_action_prob(self, state):
        raise NotImplemented

    def sample_trajectory(self, eps=None, include_first_step=False):
        raise NotImplemented

    def sample_batch(self, num_trajectories=None, include_first_step=False):
        raise NotImplemented

    def seed(self, random_seed):
        raise NotImplemented
