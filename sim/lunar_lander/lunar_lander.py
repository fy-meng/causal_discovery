import numpy as np
import gym
import scipy.special

from ..simulator import Simulator
from .deepNeuralNetwork import DeepNeuralNetwork


class LunarLander(Simulator):
    """
    The feature space is:
        0: x_pos
        1: y_pos
        2: x_vel
        3: y_vel
        4: angle
        5: angle_vel
        6: left_leg
        7: right_leg

    The action space is:
        0: do nothing
        1: fire left engine
        2: fire right engine
        3: fire bottom engine
    """
    features = sorted(['x_pos_prev', 'y_pos_prev', 'x_vel_prev', 'y_vel_prev', 'angle_prev', 'angle_vel_prev',
                       'left_leg_prev', 'right_leg_prev',
                       'x_pos', 'y_pos', 'x_vel', 'y_vel', 'angle', 'angle_vel',
                       'left_leg', 'right_leg',
                       'a_prev', 'a'])

    def __init__(self):
        self.env = gym.make('LunarLander-v2')
        self.state = self.env.reset()

        self.agent = DeepNeuralNetwork(file_name='./sim/lunar_lander/DQN_Trained.h5', verbose=True)

    def reset(self):
        self.state = self.env.reset()

    def step(self, action, eps=None):
        state, reward, done, _ = self.env.step(action)
        return state, reward, done

    def get_action_prob(self, state):
        q_values = self.agent.predict(state).squeeze()
        q_values = scipy.special.softmax(q_values)
        return q_values

    def get_action(self, state):
        """
        Returns a vector of dimension 4 where all but 1 entry is one.
        """
        q_values = self.get_action_prob(state)
        action = np.zeros_like(q_values)
        action[np.where(q_values == np.max(q_values))] = 1
        return action

    def get_action_index(self, state):
        q_values = self.get_action_prob(state)
        return np.argmax(q_values)

    def sample_trajectory(self, eps=None, include_first_step=False):
        self.reset()

        states = [self.state.copy()]
        actions = []

        done = False
        while not done:
            action_idx = self.get_action_index(self.state)
            self.state, _, done = self.step(action_idx, eps)

            action_vec = np.zeros(self.env.action_space.n)
            action_vec[action_idx] = 1

            states.append(self.state.copy())
            actions.append(action_vec)

        actions.append(self.get_action(self.state))

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
        features = ['x_pos_prev', 'y_pos_prev', 'x_vel_prev', 'y_vel_prev', 'angle_prev', 'angle_vel_prev',
                    'left_leg_prev', 'right_leg_prev',
                    'x_pos', 'y_pos', 'x_vel', 'y_vel', 'angle', 'angle_vel',
                    'left_leg', 'right_leg',
                    'a_prev', 'a']

        num_cols = len(features) + 2 * (self.env.action_space.n - 1)
        data = np.zeros((0, num_cols))
        for _ in range(num_trajectories):
            xs, us = self.sample_trajectory()
            batch = np.hstack((xs[:-1], xs[1:], us[:-1], us[1:]))
            data = np.vstack((data, batch))

        pref_sum = [self.env.action_space.n if f in ('a', 'a_prev') else 1 for f in features]
        pref_sum = list(np.cumsum(pref_sum))
        pref_sum = [0] + pref_sum

        result = np.zeros((data.shape[0], 0))
        for idx, f in enumerate(sorted(features)):
            ori_idx = features.index(f) + 1
            cols = data[:, pref_sum[ori_idx - 1]:pref_sum[ori_idx]]
            if len(cols.shape) == 1:
                cols = cols[:, np.newaxis]
            result = np.hstack((result, cols))

        return result

    def seed(self, random_seed):
        self.env.seed(random_seed)
