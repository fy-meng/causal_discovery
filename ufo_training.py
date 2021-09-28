import torch

from sim.ufo_rotate import UFORotate
from utils import Net


def select_action(state):
    # Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
    state = torch.from_numpy(state).type(torch.FloatTensor)
    state = policy(Variable(state))
    c = Categorical(state)
    action = c.sample()

    # Add log probability of our chosen action to our history
    if policy.policy_history.dim() != 0:
        policy.policy_history = torch.cat([policy.policy_history, c.log_prob(action)])
    else:
        policy.policy_history = (c.log_prob(action))
    return action


def train():
    env = UFORotate()
    agent = Net(in_channels=2, out_channels=1, layers=(64,))



