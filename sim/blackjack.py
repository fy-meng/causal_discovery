from collections import defaultdict
import os

import numpy as np
import gym
import scipy.special
from tqdm import tqdm

from .simulator import Simulator


class Blackjack(Simulator):
    """
    The feature space is:
        0: current sum of hands
        1: dealer's shown card
        2: has a usable ace or not

    The action space is:
        0: stick
        1: draw
    """
    features = sorted(['hand_prev', 'dealer_prev', 'ace_prev', 'a_prev',
                       'hand', 'dealer', 'ace', 'a'])

    def __init__(self, q_values_path='./sim/blackjack_q_values.npy'):
        self.env = gym.make('Blackjack-v0')

        if os.path.exists(q_values_path):
            self.q_values = defaultdict(lambda: np.zeros(self.env.action_space.n))
            self.q_values.update(np.load(q_values_path, allow_pickle=True).item())
            self.policy = self.make_epsilon_greedy_policy(self.q_values, 0.1, self.env.action_space.n)
        else:
            self.q_values, self.policy = self.mc_control_epsilon_greedy(self.env)
            np.save(q_values_path, np.array(dict(self.q_values)))

        self.state = self.env.reset()

    def reset(self):
        self.state = self.env.reset()

    def step(self, action, eps=None):
        state, reward, done, _ = self.env.step(action)
        return state, reward, done

    def get_action_prob(self, state):
        state = (int(state[0]), int(state[1]), bool(state[2]))

        if state[0] > 21:
            q_values = [-1, -1]
        else:
            q_values = self.q_values[state]

        q_values = scipy.special.softmax(q_values)
        return q_values

    def get_action(self, state):
        # since action space dim = 2, output 1 dim value is enough
        return self.get_action_index(state)

    def get_action_index(self, state):
        q_values = self.get_action_prob(state)
        return np.argmax(q_values)

    def sample_trajectory(self, eps=None, include_first_step=False):
        self.reset()

        states = [self.state]
        actions = []

        done = False
        while not done:
            action_idx = self.get_action_index(self.state)
            self.state, _, done = self.step(action_idx, eps)

            states.append(self.state)
            actions.append(action_idx)

        # after the game is done, append a pseudo-action
        actions.append(0)

        if include_first_step:
            states = [np.zeros_like(self.state)] + states
            actions = [0] + actions

        states = np.array(states)
        if len(states.shape) == 1:
            states = states[:, np.newaxis]
        actions = np.array(actions)

        if len(actions.shape) == 1:
            actions = actions[:, np.newaxis]

        return states, actions

    def sample_batch(self, num_trajectories=None, include_first_step=False):
        """
        :return: a batch of data sorted in name-ascending order.
        """
        unsorted_features = ['hand_prev', 'dealer_prev', 'ace_prev', 'hand', 'dealer', 'ace', 'a_prev', 'a']

        data = np.zeros((0, len(self.features)))
        for _ in range(num_trajectories):
            xs, us = self.sample_trajectory(include_first_step=include_first_step)
            batch = np.hstack((xs[:-1], xs[1:], us[:-1], us[1:]))
            data = np.vstack((data, batch))

        indices = [unsorted_features.index(f) for f in self.features]

        return data[:, indices]

    def seed(self, random_seed):
        self.env.seed(random_seed)

    @staticmethod
    def make_epsilon_greedy_policy(Q, epsilon, nA):
        """
        Creates an epsilon-greedy policy based on a given Q-function and epsilon.

        Args:
            Q: A dictionary that maps from state -> action-values.
                Each value is a numpy array of length nA (see below)
            epsilon: The probability to select a random action . float between 0 and 1.
            nA: Number of actions in the environment.

        Returns:
            A function that takes the observation as an argument and returns
            the probabilities for each action in the form of a numpy array of length nA.

        """

        def policy_fn(observation):
            # Implement this!
            props = np.ones(nA, dtype=float) * epsilon / nA
            best_action = np.argmax(Q[observation])
            props[best_action] += 1. - epsilon
            return props

        return policy_fn

    @staticmethod
    def mc_control_epsilon_greedy(env, num_episodes=500000, discount_factor=1.0, epsilon=0.1):
        """
        Monte Carlo Control using Epsilon-Greedy policies.
        Finds an optimal epsilon-greedy policy.

        Args:
            env: OpenAI gym environment.
            num_episodes: Number of episodes to sample.
            discount_factor: Gamma discount factor.
            epsilon: Chance the sample a random action. Float betwen 0 and 1.

        Returns:
            A tuple (Q, policy).
            Q is a dictionary mapping state -> action values.
            policy is a function that takes an observation as an argument and returns
            action probabilities
        """

        # Keeps track of sum and count of returns for each state
        # to calculate an average. We could use an array to save all
        # returns (like in the book) but that's memory inefficient.
        returns_sum = defaultdict(float)
        returns_count = defaultdict(float)

        # The final action-value function.
        # A nested dictionary that maps state -> (action -> action-value).
        Q = defaultdict(lambda: np.zeros(env.action_space.n))

        # The policy we're following
        policy = Blackjack.make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

        # Implement this!
        print('training blackjack agent...')
        for _ in tqdm(range(num_episodes)):
            observation = env.reset()

            episodes = []
            for i in range(100):
                props = policy(observation)
                action = np.random.choice(np.arange(len(props)), p=props)
                next_observation, reward, done, _ = env.step(action)
                episodes.append((observation, action, reward))

                if done:
                    break
                observation = next_observation

            # find the unique observation
            pairs = set([(episode[0], episode[1]) for episode in episodes])
            for (observation, action) in pairs:
                pair = (observation, action)
                # find the first occurence of the observation
                idx = episodes.index(
                    [episode for episode in episodes if episode[0] == observation and episode[1] == action][0])
                V = sum([reward[2] * discount_factor ** i for i, reward in enumerate(episodes[idx:])])

                returns_sum[pair] += V
                returns_count[pair] += 1.

                Q[observation][action] = returns_sum[pair] / returns_count[pair]

        return Q, policy
