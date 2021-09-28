class Model:
    def optimize(self, x, c):
        raise NotImplemented

    def generate(self, c, z=None):
        raise NotImplemented

    def encode(self, c, x):
        raise NotImplemented

    def train(self):
        raise NotImplemented

    def eval(self):
        raise NotImplemented

    def state_dict(self):
        raise NotImplemented

    def load_state_dict(self, state_dict):
        raise NotImplemented
