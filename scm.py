from collections import defaultdict
import json
import os
import sys

import matplotlib.pyplot as plt
import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.action_model import ActionModel
from model.bicogan import BiCoGAN
from model.linear_generator import LinearNoiseGenerator
from model.linear_regressor import LinearRegressor
from sim.bang_bang import BangBangControl
from sim.blackjack import Blackjack
from sim.lunar_lander.lunar_lander import LunarLander
from sim.simulator import Simulator
from utils import verify_output_path

plt.rcParams.update({'font.size': 14})


class SCM(nx.DiGraph):
    def __init__(self, num_latent=1, action_size=1, action_nodes=(), **kwargs):
        super().__init__(**kwargs)
        self.num_latent = num_latent
        self.action_size = action_size
        self.action_nodes = action_nodes

        self.num_cols = None
        self.indices = None

    def add_transition(self, u, transition):
        self.add_node(u, transition=transition)

    def add_model(self, u, model):
        self.add_node(u, model=model)

    def save_skeleton(self, path):
        verify_output_path(path)
        skeleton = SCM.copy_skeleton(self)
        data = json_graph.node_link_data(skeleton)
        with open(path, 'w+') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load_skeleton(path, num_latent=1, action_size=1, action_nodes=()):
        with open(path, 'r') as f:
            data = json.load(f)
            g = json_graph.node_link_graph(data)
            scm = SCM(num_latent, action_size=action_size, action_nodes=action_nodes)
            scm.add_edges_from(g.edges)
        scm.create_indices()
        return scm

    @staticmethod
    def copy_skeleton(g: nx.DiGraph):
        skeleton = SCM()
        skeleton.add_edges_from(g.edges)
        return skeleton

    def create_indices(self):
        self.num_cols = self.number_of_nodes() + len(self.action_nodes) * (self.action_size - 1)

        self.indices = [self.action_size if node in self.action_nodes else 1 for node in sorted(self.nodes)]
        self.indices = list(np.cumsum(self.indices))
        self.indices = [0] + self.indices

    def get_parent_values(self, node):
        parents = sorted(self.predecessors(node))
        parent_values = []
        for p in parents:
            v = self.nodes[p]['value']
            if not isinstance(v, list):
                if isinstance(v, np.ndarray) and v.shape != ():
                    v = list(v)
                else:
                    v = [v]
            parent_values += v

        return parent_values

    def generate(self, latent_dict=None, use_model=True, fixed_nodes=None, integer=False):
        if latent_dict is None:
            latent_dict = {}
        for node in self.nodes:
            if node in latent_dict:
                self.add_node(node, latent=latent_dict[node])
            else:
                self.add_node(node, latent=np.random.randn(self.num_latent))

        for node in nx.topological_sort(self):
            if fixed_nodes is None or node not in fixed_nodes:
                parent_values = self.get_parent_values(node)
                latent_value = [self.nodes[node]['latent']]

                if use_model:
                    parent_values = np.array(parent_values)
                    latent_value = np.array(latent_value)

                    value = self.nodes[node]['model'].generate(parent_values, latent_value)

                    if isinstance(value, torch.Tensor):
                        value = value.detach().cpu().numpy()
                    if isinstance(value, np.ndarray):
                        if integer and node not in self.action_nodes:
                            value = np.round(value).astype(int)
                        value = value.squeeze()
                    if isinstance(value, np.ndarray) and value.shape != ():
                        value = list(value)

                    self.nodes[node]['value'] = value
                else:
                    input_values = np.array(parent_values + latent_value)
                    self.add_node(node, value=self.nodes[node]['transition'](*input_values))

    def encode(self):
        for node in self.nodes:
            parent_values = self.get_parent_values(node)
            node_value = np.array([self.nodes[node]['value']])
            latent = self.nodes[node]['model'].encode(parent_values, node_value)
            if isinstance(latent, torch.Tensor):
                latent = latent.detach().cpu().numpy()
            if isinstance(latent, np.ndarray):
                latent = latent.squeeze()

            self.nodes[node]['latent'] = latent

    def optimize_bicogan(self, batch):
        """
        :param batch: the values of nodes in named-sorted order.
        """
        avg_loss_D, avg_loss_G, avg_loss_E = 0, 0, 0
        sorted_nodes = sorted(self.nodes)
        for node in nx.topological_sort(self):
            parents = sorted(self.predecessors(node))
            parent_idx = np.array([sorted_nodes.index(p) for p in parents])
            parent_values = batch[:, parent_idx]

            output_value = batch[:, sorted_nodes.index(node)]

            model: BiCoGAN = self.nodes[node]['model']
            loss_D, loss_G, loss_E = model.optimize(x=output_value, c=parent_values)
            avg_loss_D += loss_D / self.number_of_nodes()
            avg_loss_G += loss_G / self.number_of_nodes()
            avg_loss_E += loss_E / self.number_of_nodes()
        return avg_loss_D, avg_loss_G, avg_loss_E

    def optimize(self, batch):
        """
        :param batch: the values of nodes in named-ascending order.
        """
        avg_loss = 0
        sorted_nodes = sorted(self.nodes)
        for node in nx.topological_sort(self):
            parents = sorted(self.predecessors(node))
            parent_values = np.zeros((batch.shape[0], 0))
            for idx in [sorted_nodes.index(p) for p in parents]:
                cols = batch[:, self.indices[idx]:self.indices[idx + 1]]
                if len(cols.shape) == 1:
                    cols = cols[:, np.newaxis]

                parent_values = np.hstack((parent_values, cols))

            output_value = batch[:, sorted_nodes.index(node)]

            loss = self.nodes[node]['model'].optimize(x=output_value, c=parent_values)
            avg_loss += loss / self.number_of_nodes()
        return avg_loss

    def set_state(self, state):
        assert len(state) == self.num_cols
        col = 0
        for i, node in enumerate(sorted(self.nodes)):
            size = self.action_size if node in self.action_nodes else 1
            value = list(state[col:col + size])
            if len(value) == 1:
                value = value[0]
            else:
                value = list(value)
            self.nodes[node]['value'] = value
            col += size

    def get_latent_dict(self, state, sink_node):
        self.set_state(state)

        # only generate value for sink node
        self.generate(fixed_nodes=[node for node in self.nodes if node != sink_node])

        self.encode()
        latent_dict = nx.get_node_attributes(self, 'latent')
        for k, v in latent_dict.items():
            try:
                latent_dict[k] = float(v)
            except TypeError:
                latent_dict[k] = v

        return latent_dict

    def compute_saliency(self, state, start_node, sink_node, latent_dict, target_value,
                         bool_nodes=(), fixed_nodes=None, num_trials=2, perturb_amount=0.01, integer=False):
        saliency = 0.0

        model: LinearNoiseGenerator = self.nodes[start_node]['model']
        for k in range(num_trials):
            # reset the value of all nodes
            self.set_state(state)

            if start_node in bool_nodes:
                noise = 1
                self.nodes[start_node]['value'] = 1 - self.nodes[start_node]['value']
            else:
                # deterministic noise
                sign = 1 if k % 2 == 0 else -1
                noise = sign * perturb_amount
                if model.output_std is not None:
                    self.nodes[start_node]['value'] += noise * model.output_std
                else:
                    self.nodes[start_node]['value'] += noise

            # compute new values for all nodes
            if fixed_nodes is None:
                fixed_nodes = (start_node,)
            else:
                assert start_node in fixed_nodes
            self.generate(latent_dict=latent_dict, use_model=True, fixed_nodes=fixed_nodes, integer=integer)

            # get new target value and compute difference
            new_target_value = self.nodes[sink_node]['value']

            saliency += np.mean(np.abs(new_target_value - target_value)) / abs(noise) / num_trials

        return saliency

    def compute_all_saliency(self, state, sink_node,
                             bool_nodes=(), num_trials=2, perturb_amount=0.01, integer=False):
        """
        Compute saliencies for all but SINK_NODE given the state.
        """
        saliencies = dict()

        latent_dict = self.get_latent_dict(state, sink_node)

        # generate only the value for sink_node, since it's not reflected in STATE
        self.set_state(state)
        self.generate(fixed_nodes=[node for node in self.nodes if node != sink_node])
        target_value = np.array(self.nodes[sink_node]['value'])

        for node in self.nodes:
            if node != sink_node:
                saliencies[node] = self.compute_saliency(state, node, sink_node, latent_dict, target_value,
                                                         bool_nodes=bool_nodes, num_trials=num_trials,
                                                         perturb_amount=perturb_amount, integer=integer)

        return saliencies

    def compute_path_saliency(self, state, sink_node, start_nodes=None,
                              bool_nodes=(), num_trials=2, perturb_amount=0.01, integer=False):
        saliencies = dict()

        latent_dict = self.get_latent_dict(state, sink_node)

        # generate only the value for sink_node, since it's not reflected in STATE
        self.set_state(state)
        self.generate(fixed_nodes=[node for node in self.nodes if node != sink_node])
        target_value = np.array(self.nodes[sink_node]['value'])

        if start_nodes is None:
            start_nodes = [node for node in self.nodes if node != sink_node]

        for node in start_nodes:
            if node != sink_node:
                paths = [tuple(p) for p in nx.all_simple_paths(self, node, sink_node)]
                for p in paths:
                    assert len(p) >= 2
                    fixed_nodes = [node for node in self.nodes if node not in p[1:]]
                    print(p, fixed_nodes)
                    saliencies[p] = self.compute_saliency(state, node, sink_node, latent_dict, target_value,
                                                          bool_nodes=bool_nodes, fixed_nodes=fixed_nodes,
                                                          num_trials=num_trials, perturb_amount=perturb_amount,
                                                          integer=integer)

        return saliencies

    def save_models(self, path='./trained_models/scm.pth'):
        d = {node: self.nodes[node]['model'].state_dict() for node in self.nodes}
        verify_output_path(path)
        torch.save(d, path)

    def load_models(self, path='./trained_models/scm.pth'):
        state_dict = torch.load(path)
        for node in self.nodes:
            self.nodes[node]['model'].load_state_dict(state_dict[node])

    def eval(self):
        """
        Sets all nodes to evaluation mode.
        """
        for node in self.nodes:
            self.nodes[node]['model'].eval()

    def train(self):
        """
        Sets all nodes to training mode.
        :return:
        """
        for node in self.nodes:
            self.nodes[node]['model'].train()


def test(sim: Simulator, feature_nodes, action_nodes, sink_node, skeleton_path, model_path, output_dir, paths=False,
         bool_nodes=(), use_q=False, action_size=1, retrain=False, linear_regression=True, normalize=True,
         num_trajectories=100, num_epochs=200, batch_size=64, saliency_trials=100, perturb_amount=0.01, integer=False,
         seed=42, verbose=False):
    """
    Run the SCM saliency algorithm.
    :param sim: The simulator
    :param feature_nodes: The features used to plot the trajectory
    :param action_nodes: The features whose model will be replaced by sim.get_action
    :param sink_node: The feature to compute the saliency
    :param skeleton_path: The path to the SCM skeleton file
    :param model_path: The path to load the model, or save to if training
    :param output_dir: The directory that output figures will be save to
    :param action_size: The dimension of the action
    :param retrain: If True, train the model again even if model_path is not empty
    :param num_trajectories: Number of trajectories to train the model
    :param num_epochs: Number of epochs to train the model
    :param batch_size: Batch size for training
    :param seed: Random seed
    :return:
    """
    features = sim.features
    sorted_nodes = sorted(features)
    indices = [action_size if node in action_nodes else 1 for node in sorted(features)]
    indices = list(np.cumsum(indices))
    indices = [0] + indices

    # create SCM
    scm = SCM.load_skeleton(skeleton_path, num_latent=1, action_size=action_size, action_nodes=action_nodes)
    sorted_features = sorted(feature_nodes)
    reorder = [sorted_features.index(f) for f in feature_nodes]

    def action_func_q(state):
        if len(state) == len(reorder):
            state = state[reorder]
        return sim.get_action_prob(state)

    def action_func_a(state):
        if len(state) == len(reorder):
            state = state[reorder]
        return sim.get_action(state)

    for node in scm.nodes:
        if node == sink_node:
            if use_q:
                scm.nodes[node]['model'] = ActionModel(action_func_q)
            else:
                scm.nodes[node]['model'] = ActionModel(action_func_a)
        elif node in action_nodes:
            scm.nodes[node]['model'] = ActionModel(action_func_a)
        else:
            parents = tuple(scm.predecessors(node))
            cond_size = len(parents) + len([p for p in parents if p in action_nodes]) * (action_size - 1)
            if linear_regression:
                scm.nodes[node]['model'] = LinearRegressor(cond_size=cond_size, output_size=1, latent_size=1)
            else:
                scm.nodes[node]['model'] = LinearNoiseGenerator(cond_size=cond_size, output_size=1, latent_size=1)

    # train if model does not exists or RETRAIN is set to True
    # otherwise load model
    if retrain or not os.path.exists(model_path):
        print('training scm...')
        # get training data
        # features should be a sorted strings of features name correspond to the node names in the SCM
        # and the data column order should also correspond to FEATURES

        seed = np.random.randint(65536)
        sim.seed(np.random.randint(65536))
        train_data = sim.sample_batch(num_trajectories)
        print(f'training data size: {train_data.shape}')

        # record mean and var for each node in the graph
        if normalize:
            train_mean = np.mean(train_data, axis=0)
            train_std = np.std(train_data, axis=0)
            for i, feature in enumerate(features):
                parents = sorted(scm.predecessors(feature))
                parent_indices = []
                for idx in [sorted_nodes.index(p) for p in parents]:
                    parent_indices += list(range(indices[idx], indices[idx + 1]))

                model = scm.nodes[feature]['model']
                model.cond_mean = train_mean[parent_indices]
                model.cond_std = train_std[parent_indices]
                model.output_mean = train_mean[i]
                model.output_std = train_std[i]

        if linear_regression:
            loss = scm.optimize(train_data)
            print(f'loss = {loss:.4f}')
        else:
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
            scm.train()

            # training
            losses = []
            for epoch in range(num_epochs):
                loss = 0
                for n, samples in enumerate(train_loader):
                    loss = scm.optimize(samples)
                    losses.append(loss)

                if epoch % 10 == 0:
                    print(f'epoch {epoch:03d}: loss = {loss:.4f}')

        print(f'saving model to {model_path}')
        scm.save_models(model_path)

        # plot 2D or 3D nodes to check
        if verbose:
            for node in scm.nodes:
                parents = list(scm.predecessors(node))
                num_cols = sum([action_size if p in action_nodes else 1 for p in parents])
                if node not in action_nodes and 1 <= num_cols <= 2:
                    print(f'generating graph for {node}')
                    node_idx = features.index(node)

                    parents = sorted(scm.predecessors(node))
                    parent_values = np.zeros((train_data.shape[0], 0))
                    for idx in [sorted_nodes.index(p) for p in parents]:
                        cols = train_data[:, indices[idx]:indices[idx + 1]]
                        if len(cols.shape) == 1:
                            cols = cols[:, np.newaxis]

                        parent_values = np.hstack((parent_values, cols))

                    node_values = train_data[:, indices[node_idx]:indices[node_idx + 1]]

                    print(node, parents)
                    print(node_values)
                    print(parent_values)

                    est_node_values = scm.nodes[node]['model'].generate(parent_values,
                                                                        z=np.zeros((train_data.shape[0], 1)))
                    if isinstance(est_node_values, torch.Tensor):
                        est_node_values = est_node_values.detach().cpu().numpy()

                    if len(parents) == 1:
                        plt.figure(dpi=300)
                        plt.scatter(parent_values, node_values, s=1, label='real')
                        plt.scatter(parent_values, est_node_values, s=1, label='estimated')
                        plt.xlabel(parents[0])
                        plt.ylabel(node)
                    elif len(parents) == 2:
                        fig = plt.figure(dpi=300)
                        ax = fig.add_subplot(projection='3d')
                        ax.scatter(parent_values[:, 0], parent_values[:, 1], node_values, s=1, label='real')
                        ax.scatter(parent_values[:, 0], parent_values[:, 1], est_node_values, s=1, label='estimated')
                        ax.set_xlabel(parents[0])
                        ax.set_ylabel(parents[1])
                        ax.set_zlabel(node)
                    plt.legend()
                    output_file = os.path.join(output_dir, f'scm_{node}.png')
                    verify_output_path(output_file)
                    plt.savefig(output_file, bbox_inches='tight')
                    plt.clf()

    else:
        print(f'loading model from {model_path}')
        scm.load_models(model_path)

    if verbose:
        for node in sorted(scm.nodes):
            if not isinstance(scm.nodes[node]['model'], ActionModel):
                print(f"{node}:\n"
                      f"\tcond_mean={scm.nodes[node]['model'].cond_mean}\n"
                      f"\tcond_std={scm.nodes[node]['model'].cond_std}\n"
                      f"\toutput_mean={scm.nodes[node]['model'].output_mean}\n"
                      f"\toutput_std={scm.nodes[node]['model'].output_std}\n"
                      f"\tvar={scm.nodes[node]['model'].var}")

    # testing
    np.random.seed(seed)
    sim.seed(seed)
    scm.eval()

    test_data = sim.sample_batch(1, include_first_step=True)

    # compute saliency
    saliency_dict = defaultdict(lambda: [])
    for t in tqdm(range(len(test_data))):
        if paths:
            # TODO: start nodes
            saliencies = scm.compute_path_saliency(test_data[t], sink_node, start_nodes=['a_prev'],
                                                   num_trials=saliency_trials, perturb_amount=perturb_amount,
                                                   integer=integer, bool_nodes=bool_nodes)
        else:
            saliencies = scm.compute_all_saliency(test_data[t], sink_node,
                                                  num_trials=saliency_trials, perturb_amount=perturb_amount,
                                                  integer=integer, bool_nodes=bool_nodes)
        for feature, value in saliencies.items():
            saliency_dict[feature].append(value)

    if verbose:
        print('test_data:')
        print(test_data)

        # plot trajectory
        print('plotting trajectory...')
        plt.figure(dpi=300)
        for node in feature_nodes:
            idx = features.index(node)
            plt.plot(test_data[:, indices[idx]:indices[idx + 1]], label=node)
        plt.legend()
        output_file = os.path.join(output_dir, 'trajectory.png')
        verify_output_path(output_file)
        plt.savefig(output_file, bbox_inches='tight')
        plt.clf()

        # plot saliency
        print('plotting saliency...')
        plt.figure(dpi=300)
        for feature, saliencies in saliency_dict.items():
            plt.plot(saliencies, label=feature)
        plt.legend()
        output_file = os.path.join(output_dir, 'saliency.png')
        verify_output_path(output_file)
        plt.savefig(output_file, bbox_inches='tight')
        plt.clf()

    # save state and saliency to npz
    verify_output_path(os.path.join(output_dir, 'trajectory.npy'))
    np.save(os.path.join(output_dir, 'trajectory.npy'), test_data)
    np.savez(os.path.join(output_dir, 'saliency_dict.npz'), **saliency_dict)


def test_blackjack_extended():
    num_step = 5

    sim = Blackjack()
    action_nodes = [f'a_{i}' for i in range(num_step)]
    sink_node = 'a_4'
    feature_nodes = ('hand', 'dealer', 'ace')
    action_size = 1

    skeleton_path_ori = './skeletons/blackjack_scm.json'
    skeleton_path = './skeletons/blackjack_extended_scm.json'

    model_path = './trained_models/blackjack_scm.pth'

    # create SCM for the one step graph
    scm_ori = SCM.load_skeleton(skeleton_path_ori, num_latent=1, action_size=action_size, action_nodes=action_nodes)
    sorted_features = sorted(feature_nodes)
    reorder = [sorted_features.index(f) for f in feature_nodes]
    for node in scm_ori.nodes:
        if node == 'a':
            scm_ori.nodes[node]['model'] = ActionModel(lambda state: sim.get_action_prob(state[reorder]))
        elif node == 'a_prev':
            scm_ori.nodes[node]['model'] = ActionModel(lambda state: sim.get_action(state[reorder]))
        else:
            parents = tuple(scm_ori.predecessors(node))
            cond_size = len(parents) + len([p for p in parents if p in action_nodes]) * (action_size - 1)
            scm_ori.nodes[node]['model'] = LinearNoiseGenerator(cond_size=cond_size, output_size=1, latent_size=1)

    # load model
    print(f'loading model from {model_path}')
    scm_ori.load_models(model_path)

    # create SCM for multi step graph
    scm = SCM.load_skeleton(skeleton_path, num_latent=1, action_size=action_size, action_nodes=action_nodes)
    sorted_features = sorted(feature_nodes)
    reorder = [sorted_features.index(f) for f in feature_nodes]
    for node in scm.nodes:
        if node == sink_node:
            scm.nodes[node]['model'] = ActionModel(lambda state: sim.get_action_prob(state[reorder]))
        elif node in action_nodes:
            scm.nodes[node]['model'] = ActionModel(lambda state: sim.get_action(state[reorder]))
        else:
            parents = tuple(scm.predecessors(node))
            cond_size = len(parents) + len([p for p in parents if p in action_nodes]) * (action_size - 1)
            scm.nodes[node]['model'] = LinearNoiseGenerator(cond_size=cond_size, output_size=1, latent_size=1)

    # transfer model
    for i in range(2, num_step):
        for f in ('hand', 'ace', 'dealer'):
            scm.nodes[f'{f}_{i}']['model'] = scm_ori.nodes[f]['model']
    del scm_ori

    # testing
    scm.eval()
    states, actions = sim.sample_trajectory()
    test_data = np.hstack((states, actions))
    test_data = test_data[:-1]
    test_data = test_data.flatten()

    print(test_data.shape)

    features = []
    for i in range(num_step):
        for j, f in enumerate(('hand', 'dealer', 'ace', 'a')):
            features.append(f'{f}_{i}')
    sorted_features = sorted(features)

    idx = [features.index(f) for f in sorted_features]
    state = (test_data[idx])

    bool_nodes = [f'a_{i}' for i in range(5)] + [f'ace_{i}' for i in range(5)]
    saliency = scm.compute_all_saliency(state, sink_node, bool_nodes=bool_nodes,
                                        num_trials=2, perturb_amount=1, integer=True)

    plt.figure(dpi=300)
    labels = ['Hand_t', 'Dealer_t', 'Ace_t', 'Action_t']
    for f, label in zip(['hand', 'dealer', 'ace', 'a'], labels):
        s = [saliency[f'{f}_{i}'] for i in range(num_step if f != 'a' else num_step - 1)]
        if f != 'a':
            plt.plot(np.arange(2, 6), s[1:], marker='o', label=label)
        else:
            plt.plot(np.arange(2, 5), s[1:], marker='o', label=label)
    plt.legend(fontsize=16)
    plt.xlabel('Time step t', fontsize=16)
    plt.ylabel('Importance on A_5', fontsize=16)
    plt.xticks([2, 3, 4, 5])
    plt.savefig('./blackjack_multistep.png', bbox_inches='tight')
    plt.clf()

    # saliencies = {'hand_1': [], 'dealer_1': [], 'ace_1': [], 'a_1': []}
    # for i in range(1, num_step):
    #     ss = scm.compute_all_saliency(state, sink_node=f'a_{i}', num_trials=2, perturb_amount=1, integer=True)
    #     for f in saliencies.keys():
    #         if f in ss.keys():
    #             saliencies[f].append(ss[f])
    # for f, s in saliencies.items():
    #     if f == 'a_1':
    #         plt.plot([2, 3, 4], s, label=f)
    #     else:
    #         plt.plot([1, 2, 3, 4], s, label=f)
    # plt.legend()
    # plt.title('importance of step 1 features on actions 1-4')
    # plt.show()


def test_uncertainty(num_tries, idx, sim: Simulator, feature_nodes, action_nodes, sink_node, skeleton_path, model_path,
                     output_dir, paths=False, bool_nodes=(), use_q=False, action_size=1, retrain=True,
                     linear_regression=True, normalize=True, num_trajectories=100, num_epochs=200, batch_size=64,
                     saliency_trials=100, perturb_amount=0.01, integer=False, seed=42):
    p, ext = os.path.splitext(model_path)

    ranges = range(num_tries) if idx is None else [idx]
    for t in ranges:
        curr_output_dir = os.path.join(output_dir, f'{t}')
        curr_model_path = p + f'_{t}' + ext
        test(sim, feature_nodes, action_nodes, sink_node, skeleton_path, curr_model_path, curr_output_dir, paths,
             bool_nodes, use_q, action_size, retrain, linear_regression, normalize, num_trajectories, num_epochs,
             batch_size, saliency_trials, perturb_amount, integer, seed, False)


if __name__ == '__main__':
    # run bang-bang control
    # test(sim=BangBangControl(), feature_nodes=('x', 'v', 's', 'a'), action_nodes=('a_prev', 'a'), sink_node='a',
    #      skeleton_path='./skeletons/bang_bang_scm.json', model_path='./trained_models/bang_bang_scm.pth',
    #      output_dir='./output/bang_bang_1/', retrain=False, linear_regression=True,
    #      saliency_trials=2, perturb_amount=0.1, num_epochs=50, verbose=True)

    # run lunar lander
    # test(sim=LunarLander(),
    #      feature_nodes=('x_pos', 'y_pos', 'x_vel', 'y_vel', 'angle', 'angle_vel', 'left_leg', 'right_leg'),
    #      action_size=4, action_nodes=('a_prev', 'a'), sink_node='a',
    #      bool_nodes=('left_leg', 'right_leg', 'left_leg_prev', 'right_leg_prev'),
    #      skeleton_path='./skeletons/lunar_lander_scm.json', model_path='./trained_models/lunar_lander_scm.pth',
    #      output_dir='./output/lunar_lander_1/', retrain=False, linear_regression=True,
    #      num_trajectories=100, saliency_trials=2, perturb_amount=0.01, verbose=True)

    # run blackjack action-based
    # test(sim=Blackjack(),
    #      feature_nodes=('hand', 'dealer', 'ace'),
    #      action_size=1, action_nodes=('a_prev', 'a'), sink_node='a', bool_nodes=('a', 'a_prev', 'ace', 'ace_prev'),
    #      skeleton_path='./skeletons/blackjack_scm.json', model_path='./trained_models/blackjack_scm.pth',
    #      output_dir='./output/blackjack_a/', retrain=False, linear_regression=False, normalize=False,
    #      num_trajectories=50000, num_epochs=50, saliency_trials=2, perturb_amount=1, integer=True, verbose=False)

    # run blackjack q-based
    # test(sim=Blackjack(),
    #      feature_nodes=('hand', 'dealer', 'ace'), use_q=True,
    #      action_size=1, action_nodes=('a_prev', 'a'), sink_node='a', bool_nodes=('a', 'a_prev', 'ace', 'ace_prev'),
    #      skeleton_path='./skeletons/blackjack_scm.json', model_path='./trained_models/blackjack_scm.pth',
    #      output_dir='./output/blackjack/', retrain=False, linear_regression=False, normalize=False,
    #      num_trajectories=50000, num_epochs=50, saliency_trials=2, perturb_amount=1, integer=True, verbose=False)

    # run multi-step blackjack
    # TODO: this is jank
    # test_blackjack_extended()

    # run path-specific blackjack
    # test(sim=Blackjack(), paths=True,
    #      feature_nodes=('hand', 'dealer', 'ace'), use_q=True,
    #      action_size=1, action_nodes=('a_prev', 'a'), sink_node='a', bool_nodes=('a', 'a_prev', 'ace', 'ace_prev'),
    #      skeleton_path='./skeletons/blackjack_scm.json', model_path='./trained_models/blackjack_scm.pth',
    #      output_dir='./output/blackjack_path/', retrain=False, linear_regression=False, normalize=False,
    #      num_trajectories=50000, num_epochs=50, saliency_trials=2, perturb_amount=1, integer=True, verbose=False)

    # run blackjack uncertainty
    if len(sys.argv) > 1:
        idx = int(sys.argv[-1])
    else:
        idx = None
    test_uncertainty(10, idx, sim=Blackjack(),
                     feature_nodes=('hand', 'dealer', 'ace'), use_q=True,
                     action_size=1, retrain=False, action_nodes=('a_prev', 'a'), sink_node='a',
                     bool_nodes=('a', 'a_prev', 'ace', 'ace_prev'),
                     skeleton_path='./skeletons/blackjack_scm.json', model_path='./trained_models/blackjack_scm.pth',
                     output_dir='./output/blackjack_uncertainty/', linear_regression=False, normalize=False,
                     num_trajectories=10000, num_epochs=50, saliency_trials=2, perturb_amount=1,
                     integer=True)

    # run bang-bang uncertainty
    # if len(sys.argv) > 1:
    #     idx = int(sys.argv[-1])
    # else:
    #     idx = None
    # test_uncertainty(10, idx, sim=BangBangControl(noise=True), feature_nodes=('x', 'v', 's', 'a'),
    #                  action_nodes=('a_prev', 'a'),
    #                  sink_node='a', skeleton_path='./skeletons/bang_bang_scm.json',
    #                  model_path='./trained_models/bang_bang_scm.pth', num_trajectories=1,
    #                  output_dir='./output/bang_bang_uncertainty/', retrain=False, linear_regression=True,
    #                  saliency_trials=2, perturb_amount=0.1, num_epochs=50)
