import json

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from networkx.readwrite import json_graph
import numpy as np
import torch

from bicogan import BiCoGAN, LinearNoiseGenerator, LinearNoiseRegressor
from utils import verify_output_path


class SCM(nx.DiGraph):
    def __init__(self, num_latent=1, **kwargs):
        super().__init__(**kwargs)
        self.num_latent = num_latent

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
    def load_skeleton(path, num_latent=1):
        with open(path, 'r') as f:
            data = json.load(f)
            g = json_graph.node_link_graph(data)
            scm = SCM(num_latent)
            scm.add_edges_from(g.edges)
        return scm

    @staticmethod
    def copy_skeleton(g: nx.DiGraph):
        skeleton = SCM()
        skeleton.add_edges_from(g.edges)
        return skeleton

    def generate(self, latent_dict=None, use_model=False):
        if latent_dict is None:
            latent_dict = {}
        for node in self.nodes:
            if node in latent_dict:
                self.add_node(node, latent=latent_dict[node])
            else:
                self.add_node(node, latent=np.random.randn(self.num_latent))

        for node in nx.topological_sort(self):
            parents = sorted(self.predecessors(node))
            parent_values = [self.nodes[p]['value'] for p in parents]
            latent_value = [self.nodes[node]['latent']]
            if use_model:
                parent_values = np.array(parent_values)
                latent_value = np.array(latent_value)

                value = self.nodes[node]['model'].generate(parent_values, latent_value).detach().cpu().numpy().squeeze()
                self.add_node(node, value=value)
            else:
                input_values = np.array(parent_values + latent_value)
                self.add_node(node, value=self.nodes[node]['transition'](*input_values))

    def encode(self):
        for node in self.nodes:
            parents = sorted(self.predecessors(node))
            parent_values = np.array([self.nodes[p]['value'] for p in parents])
            node_value = np.array([self.nodes[node]['value']])
            latent = self.nodes[node]['model'].encode(parent_values, node_value)
            if isinstance(latent, torch.Tensor):
                latent = latent.detach().cpu().numpy()
            latent = latent.squeeze()
            self.add_node(node, latent=latent)

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
        :param batch: the values of nodes in named-sorted order.
        """
        avg_loss = 0
        sorted_nodes = sorted(self.nodes)
        for node in nx.topological_sort(self):
            parents = sorted(self.predecessors(node))
            parent_idx = np.array([sorted_nodes.index(p) for p in parents], dtype=int)
            parent_values = batch[:, parent_idx]

            output_value = batch[:, sorted_nodes.index(node)]

            loss = self.nodes[node]['model'].optimize(x=output_value, c=parent_values)
            avg_loss += loss / self.number_of_nodes()
        return avg_loss

    def save_models(self, path='./trained_models/scm.pth'):
        d = {node: self.nodes[node]['model'].state_dict() for node in self.nodes}
        verify_output_path(path)
        torch.save(d, path)

    def load_models(self, path='./trained_models/scm.pth'):
        state_dict = torch.load(path)
        for node in self.nodes:
            self.nodes[node]['model'].load_state_dict(state_dict[node])

    def eval(self):
        for node in self.nodes:
            self.nodes[node]['model'].eval()

    def train(self):
        for node in self.nodes:
            self.nodes[node]['model'].train()


def test(num_epochs=300):
    # create ground truth model
    ground_truth = SCM()
    ground_truth.add_nodes_from([('s1', {'transition': lambda us1: us1}),
                                 ('s2', {'transition': lambda s1, us2: 2 * s1 + us2}),
                                 ('s3', {'transition': lambda s1, us3: 4 * s1 + us3}),
                                 ('s4', {'transition': lambda s3, us4: 2 * s3 * us4}),
                                 ('a', {'transition': lambda s1, s3, ua: 2 * s1 + s3 + ua}),
                                 ('r', {'transition': lambda s4, a, ur: s4 + a + ur})])
    ground_truth.add_edges_from([('s1', 's2'), ('s1', 's3'), ('s1', 'a'),
                                 ('s3', 's4'), ('s3', 'a'),
                                 ('s4', 'r'),
                                 ('a', 'r')])
    ground_truth.generate()

    # create trainable BiCoGAN SCM
    scm = SCM.copy_skeleton(ground_truth)
    for node in scm.nodes:
        num_parents = len(tuple(scm.predecessors(node)))
        scm.add_model(node, BiCoGAN(cond_size=num_parents, output_size=1, latent_size=1))
        scm.add_transition(node, lambda x, z: scm.nodes[node]['model'].generate(x, z))

    # create training data
    train_data_length = 1024
    train_data = np.zeros((train_data_length, scm.number_of_nodes()))
    for i in range(train_data_length):
        ground_truth.generate()
        value_dict = nx.get_node_attributes(ground_truth, 'value')
        train_data[i] = [value_dict[node] for node in sorted(ground_truth.nodes)]
    train_data = torch.tensor(train_data).float()
    train_labels = torch.zeros(train_data_length)
    train_set = [
        (train_data[i], train_labels[i]) for i in range(train_data_length)
    ]

    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # train
    loss_ds, loss_gs, loss_es = [], [], []
    for epoch in range(num_epochs):
        loss_D, loss_G, loss_E = 0, 0, 0
        for n, (samples, _) in enumerate(train_loader):
            loss_D, loss_G, loss_E = scm.optimize(samples)

            loss_ds.append(loss_D)
            loss_gs.append(loss_G)
            loss_es.append(loss_E)

        if epoch % 10 == 0:
            print(f'epoch {epoch:03d}: loss_D = {loss_D:.4f}, loss_G = {loss_G:.4f}, loss_E = {loss_E:.4f}')

    scm.save_models()


def visualize():
    scm = SCM.load_skeleton('./skeletons/test_scm.json')
    for node in scm.nodes:
        num_parents = len(tuple(scm.predecessors(node)))
        scm.add_model(node, BiCoGAN(cond_size=num_parents, output_size=1, latent_size=1))
        scm.nodes[node]['model'].eval()

    scm.load_models('./trained_models/scm.pth')

    scm.generate(use_model=True)

    labels = {node: f'{node}: {scm.nodes[node]["value"]:.2f}' for node in scm.nodes}
    nx.draw(scm, pos=graphviz_layout(scm), labels=labels, node_size=2500)
    plt.show()


def test_clinic(num_epochs=300):
    def generate_batch(batch_size):
        # features: age (20-80), weight (~40-90), severity (0-1), outcome (0 or 1)
        # action: procedure (0 or 1)
        # action only consider severity
        age = np.random.uniform(20, 80, size=batch_size)
        weight = np.where(age <= 40,
                          0.25 * age + 60,
                          -0.25 * age + 80
                          ) + np.random.uniform(-20, 20, size=batch_size)
        normalized_age = (age - 20) / 60
        normalized_weight = (weight - 40) / 50
        severity = (3 * normalized_age + 1 * normalized_weight + 6 * np.random.uniform(0, 1, size=batch_size)) / 10
        treatment = severity > 0.5
        outcome = np.where(treatment,
                           (2 * (1 - normalized_age) + 1 * np.random.randn(batch_size)) / 3,
                           (3 * severity + 1 * np.random.randn(batch_size)) / 4
                           ) > 0.5
        # return a matrix of shape (batch_size, 5)
        # column are in sorted order, i.e., age, outcome, severity, treatment, weight
        return np.vstack((age, outcome, severity, treatment, weight)).T

    # use gaussian noise instead
    def generate_batch(batch_size):
        # features: age (20-80), weight (~40-90), severity (0-1), outcome (0 or 1)
        # action: procedure (0 or 1)
        # action only consider severity
        age = np.random.uniform(20, 80, size=batch_size)
        weight = np.where(age <= 40,
                          0.25 * age + 60,
                          -0.25 * age + 80
                          ) + 6 * np.random.randn(batch_size)
        weight = np.clip(weight, 40, 80)
        normalized_age = (age - 20) / 60
        normalized_weight = (weight - 40) / 50
        severity = (3 * normalized_age + 1 * normalized_weight + 6 * np.random.randn(batch_size)) / 10
        treatment = severity > 0.5
        outcome = np.where(treatment,
                           (2 * (1 - normalized_age) + 1 * np.random.randn(batch_size)) / 3,
                           (3 * severity + 1 * np.random.randn(batch_size)) / 4
                           ) > 0.5
        # return a matrix of shape (batch_size, 5)
        # column are in sorted order, i.e., age, outcome, severity, treatment, weight
        return np.vstack((age, outcome, severity, treatment, weight)).T

    # create trainable BiCoGAN SCM
    scm = SCM.load_skeleton('skeletons/clinic_scm.json', num_latent=1)
    for node in scm.nodes:
        num_parents = len(tuple(scm.predecessors(node)))
        scm.add_model(node, BiCoGAN(cond_size=num_parents, output_size=1, latent_size=4))

    # create training data
    train_data_length = 4096
    train_data = generate_batch(train_data_length)
    train_data = torch.tensor(train_data).float()
    train_labels = torch.zeros(train_data_length)
    train_set = [
        (train_data[i], train_labels[i]) for i in range(train_data_length)
    ]

    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # train
    loss_ds, loss_gs, loss_es = [], [], []
    for epoch in range(num_epochs):
        loss_D, loss_G, loss_E = 0, 0, 0
        for n, (samples, _) in enumerate(train_loader):
            loss_D, loss_G, loss_E = scm.optimize(samples)

            loss_ds.append(loss_D)
            loss_gs.append(loss_G)
            loss_es.append(loss_E)

        if epoch % 10 == 0:
            print(f'epoch {epoch:03d}: loss_D = {loss_D:.4f}, loss_G = {loss_G:.4f}, loss_E = {loss_E:.4f}')

    scm.save_models('./trained_models/clinic_scm.pth')


def visualize_clinic():
    scm = SCM.load_skeleton('./skeletons/clinic_scm.json', num_latent=4)
    for node in scm.nodes:
        num_parents = len(tuple(scm.predecessors(node)))
        scm.add_model(node, BiCoGAN(cond_size=num_parents, output_size=1, latent_size=4))
        scm.nodes[node]['model'].eval()

    scm.load_models('./trained_models/clinic_scm.pth')

    scm.generate(use_model=True)

    labels = {node: f'{node}: {scm.nodes[node]["value"]:.2f}' for node in scm.nodes}
    nx.draw(scm, pos=graphviz_layout(scm), labels=labels, node_size=2500)
    plt.show()
    plt.clf()

    ages = []
    weights = []
    for _ in range(1000):
        scm.generate(use_model=True)
        ages.append(scm.nodes['age']['value'])
        weights.append(scm.nodes['weight']['value'])
    ages = np.array(ages)
    weights = np.array(weights)
    plt.scatter(ages, weights, label='generated')

    def generate_batch(batch_size):
        # features: age (20-80), weight (~40-90), severity (0-1), outcome (0 or 1)
        # action: procedure (0 or 1)
        # action only consider severity
        age = np.random.uniform(20, 80, size=batch_size)
        weight = np.where(age <= 40,
                          0.25 * age + 60,
                          -0.25 * age + 80
                          ) + np.random.uniform(-20, 20, size=batch_size)
        normalized_age = (age - 50) / 5
        normalized_weight = (weight - 60) / 3.33
        severity = (3 * normalized_age + 1 * normalized_weight + 6 * np.random.randn(batch_size)) / 10
        treatment = severity > 0
        outcome = np.where(treatment,
                           (2 * -normalized_age + 1 * np.random.randn(batch_size)) / 3,
                           (3 * -severity + 1 * np.random.randn(batch_size)) / 4
                           ) > 0
        # return a matrix of shape (batch_size, 5)
        # column are in sorted order, i.e., age, outcome, severity, treatment, weight
        return np.vstack((age, outcome, severity, treatment, weight)).T

    # use gaussian noise instead
    def generate_batch(batch_size):
        # features: age (20-80), weight (~40-90), severity (0-1), outcome (0 or 1)
        # action: procedure (0 or 1)
        # action only consider severity
        age = np.random.uniform(20, 80, size=batch_size)
        weight = np.where(age <= 40,
                          0.25 * age + 60,
                          -0.25 * age + 80
                          ) + 6 * np.random.randn(batch_size)
        weight = np.clip(weight, 40, 80)
        normalized_age = (age - 20) / 60
        normalized_weight = (weight - 40) / 50
        severity = (3 * normalized_age + 1 * normalized_weight + 6 * np.random.randn(batch_size)) / 10
        treatment = severity > 0.5
        outcome = np.where(treatment,
                           (2 * (1 - normalized_age) + 1 * np.random.randn(batch_size)) / 3,
                           (3 * severity + 1 * np.random.randn(batch_size)) / 4
                           ) > 0.5
        # return a matrix of shape (batch_size, 5)
        # column are in sorted order, i.e., age, outcome, severity, treatment, weight
        return np.vstack((age, outcome, severity, treatment, weight)).T

    truth = generate_batch(1000)
    plt.scatter(truth[:, 0], truth[:, -1], label='ground truth')
    plt.legend()
    plt.show()


def test_obstacle(num_trajectories=100, num_epochs=200):
    # use gaussian noise instead
    def generate_data(num_trajectories):
        # features: x (location), s (measured distance to obstacle), v (velocity)
        # action: a (acceleration)
        # action only consider s
        data = []

        for _ in range(num_trajectories):
            d = 6 * np.random.randn() + 50
            d = np.clip(d, 20, None)
            a_max = 0.5
            v_max = 2

            x = 0
            s = d - x + np.random.randn()
            v = 0
            a = a_max if s >= (v_max ** 2) / (2 * a_max) else -a_max
            t = np.random.uniform(0.5, 1)

            prev_state = [a, s, t, v, x]

            while x < d:
                x += v * t + 0.5 * a * (t ** 2)
                v += a * t
                v = min(v, v_max)
                s = d - x + np.random.randn()
                a = a_max if s >= (v_max ** 2) / (2 * a_max) else -a_max

                state = [a, s, t, v, x]
                data.append(prev_state + state)
                prev_state = state

        return np.array(data)

    train_data = generate_data(num_trajectories)
    train_data = np.hstack((train_data[:, :7], train_data[:, 8:]))
    features = ['a_prev', 's_prev', 't_prev', 'v_prev', 'x_prev', 'a', 's', 'v', 'x']
    idx = np.argsort(features)
    train_data = train_data[:, idx]
    train_data = torch.tensor(train_data).float()
    train_data_length = train_data.shape[0]
    train_labels = torch.zeros(train_data_length)
    train_set = [
        (train_data[i], train_labels[i]) for i in range(train_data_length)
    ]

    print(train_data.shape)

    # create trainable BiCoGAN SCM
    scm = SCM.load_skeleton('skeletons/obstacle_extended_scm.json', num_latent=1)
    for node in scm.nodes:
        num_parents = len(tuple(scm.predecessors(node)))
        scm.add_model(node, LinearNoiseGenerator(cond_size=num_parents, output_size=1, latent_size=1))

    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # train
    losses = []
    for epoch in range(num_epochs):
        loss = 0
        for n, (samples, _) in enumerate(train_loader):
            loss = scm.optimize(samples)
            losses.append(loss)

        if epoch % 10 == 0:
            print(f'epoch {epoch:03d}: loss = {loss:.4f}')

    scm.save_models('./trained_models/obstacle_extended_scm.pth')


def visualize_obstacle():
    np.random.seed(42)

    # create trainable BiCoGAN SCM
    scm = SCM.load_skeleton('skeletons/obstacle_extended_scm.json', num_latent=1)
    for node in scm.nodes:
        num_parents = len(tuple(scm.predecessors(node)))
        scm.add_model(node, LinearNoiseGenerator(cond_size=num_parents, output_size=1, latent_size=1))
    scm.load_models('./trained_models/obstacle_extended_scm.pth')

    scm.eval()

    for node in sorted(scm.nodes):
        print(node, scm.nodes[node]['model'].mean, scm.nodes[node]['model'].var)

    def generate_data(num_trajectories):
        # features: x (location), s (measured distance to obstacle), v (velocity)
        # action: a (acceleration)
        # action only consider s
        data = []

        for _ in range(num_trajectories):
            d = 6 * np.random.randn() + 50
            d = np.clip(d, 20, None)
            a_max = 0.5
            v_max = 2

            x = 0
            s = d - x + np.random.randn()
            v = 0
            a = a_max if s >= (v_max ** 2) / (2 * a_max) else -a_max
            t = np.random.uniform(0.5, 1)

            prev_state = [a, s, t, v, x]

            while x < d:
                x += v * t + 0.5 * a * (t ** 2)
                v += a * t
                v = min(v, v_max)
                s = d - x + np.random.randn()
                a = a_max if s >= (v_max ** 2) / (2 * a_max) else -a_max

                state = [a, s, t, v, x]
                data.append(prev_state + state)
                prev_state = state

        return np.array(data)

    data = generate_data(1)
    data = np.hstack((data[:, :7], data[:, 8:]))
    features = ['a_prev', 's_prev', 't_prev', 'v_prev', 'x_prev', 'a', 's', 'v', 'x']
    # find the last step before changing acceleration
    idx = np.argmax(data[:, 5] < 0)
    state = data[idx]
    # state = data[np.random.randint(data.shape[0])]
    for feature, value in zip(features, state):
        scm.add_node(feature, value=value)
    print(nx.get_node_attributes(scm, 'value'))

    pos = {'t_prev': (10, 50), 'v_prev': (10, 40), 'v': (20, 40),
           'x_prev': (10, 30), 'x': (20, 30), 's_prev': (10, 20), 's': (20, 20),
           'a_prev': (10, 10), 'a': (20, 10)}
    labels = {node: f'{node}: {scm.nodes[node]["value"]:.2f}' for node in scm.nodes}
    nx.draw(scm, pos=pos, labels=labels, node_size=2500)
    plt.savefig('./output/obstacle_extended_state.png')
    plt.clf()

    scm.encode()
    labels = {node: f'{node}: {scm.nodes[node]["latent"]:.2f}' for node in scm.nodes}
    nx.draw(scm, pos=pos, labels=labels, node_size=2500)
    plt.savefig('./output/obstacle_extended_latent.png')
    plt.clf()

    saliency_trials = 100
    perturb_amount = 1
    saliency = {node: 0 for node in scm.nodes if node != 'a'}
    latent_dict = nx.get_node_attributes(scm, 'latent')
    latent_dict = {k: float(v) for k, v in latent_dict.items()}

    for node in scm.nodes:
        if node != 'a':
            for _ in range(saliency_trials):
                perturbed_latent_dict = latent_dict.copy()
                perturbed_latent_dict[node] += perturb_amount * np.random.randn()
                # perturbed_latent_dict[node] = np.random.randn()
                scm.generate(latent_dict=perturbed_latent_dict, use_model=True)
                a_max = 0.5
                v_max = 2
                new_a = a_max if scm.nodes['s']['value'] >= (v_max ** 2) / (2 * a_max) else -a_max
                saliency[node] += int(state[features.index('a')] != new_a) / saliency_trials

    labels = {node: f'{node}: {saliency[node]:.2f}' for node in scm.nodes if node != 'a'}
    labels['a'] = 'a'
    nx.draw(scm, pos=pos, labels=labels, node_size=2500)
    plt.savefig('./output/obstacle_extended_saliency.png')
    plt.clf()


def test_obstacle_linear(num_trajectories=100):
    # use gaussian noise instead
    def generate_data(num_trajectories):
        # features: x (location), s (measured distance to obstacle), v (velocity)
        # action: a (acceleration)
        # action only consider s
        data = []

        for _ in range(num_trajectories):
            d = 6 * np.random.randn() + 50
            d = np.clip(d, 20, None)
            a_max = 0.5
            v_max = 2

            x = 0
            s = d - x + np.random.randn()
            v = 0
            a = a_max if s >= (v_max ** 2) / (2 * a_max) else -a_max
            t = np.random.uniform(0.5, 1)

            prev_state = [a, s, t, v, x]

            while x < d:
                x += v * t + 0.5 * a * (t ** 2)
                v += a * t
                v = min(v, v_max)
                s = d - x + np.random.randn()
                a = a_max if s >= (v_max ** 2) / (2 * a_max) else -a_max

                state = [a, s, t, v, x]
                data.append(prev_state + state)
                prev_state = state

        return np.array(data)

    train_data = generate_data(num_trajectories)
    train_data = np.hstack((train_data[:, :7], train_data[:, 8:]))
    features = ['a_prev', 's_prev', 't_prev', 'v_prev', 'x_prev', 'a', 's', 'v', 'x']
    idx = np.argsort(features)
    train_data = train_data[:, idx]
    print(train_data.shape)

    # create trainable BiCoGAN SCM
    scm = SCM.load_skeleton('skeletons/obstacle_extended_scm.json', num_latent=1)
    for node in scm.nodes:
        num_parents = len(tuple(scm.predecessors(node)))
        scm.add_model(node, LinearNoiseRegressor(cond_size=num_parents, output_size=1, latent_size=1))

    # train
    scm.optimize(train_data)
    scm.save_models('./trained_models/obstacle_extended_linear_scm.pth')

    # testing
    np.random.seed(42)

    data = generate_data(1)
    data = np.hstack((data[:, :7], data[:, 8:]))
    state = data[np.random.choice(data.shape[0])]
    for feature, value in zip(features, state):
        scm.add_node(feature, value=value)

    pos = {'t_prev': (10, 50), 'v_prev': (10, 40), 'v': (20, 40),
           'x_prev': (10, 30), 'x': (20, 30), 's_prev': (10, 20), 's': (20, 20),
           'a_prev': (10, 10), 'a': (20, 10)}
    labels = {node: f'{node}: {scm.nodes[node]["value"]:.2f}' for node in scm.nodes}
    nx.draw(scm, pos=pos, labels=labels, node_size=2500)
    plt.savefig('./output/obstacle_extended_linear_state.png')
    plt.clf()

    scm.encode()
    labels = {node: f'{node}: {scm.nodes[node]["latent"]:.2f}' for node in scm.nodes}
    nx.draw(scm, pos=pos, labels=labels, node_size=2500)
    plt.savefig('./output/obstacle_extended_linear_latent.png')
    plt.clf()

    data = generate_data(100)

    s = np.linspace(0, 80, 100)[:, np.newaxis]
    a_estimated = scm.nodes['a']['model'].generate(s)
    plt.plot(s, a_estimated, label='estimated')
    plt.scatter(data[:, 6], data[:, 5], label='ground truth', c='orange')
    plt.legend()
    plt.show()
    plt.clf()

    x = np.linspace(0, 80, 100)[:, np.newaxis]
    s_estimated = scm.nodes['s']['model'].generate(x)
    plt.plot(x, s_estimated, label='estimated')
    plt.scatter(data[:, 9], data[:, 6], label='ground truth', c='orange')
    plt.legend()
    plt.show()

    saliency_trials = 100
    perturb_amount = 0.2
    saliency = {node: 0 for node in scm.nodes if node != 'a'}

    for node in scm.nodes:
        if node != 'a':
            for _ in range(saliency_trials):
                scm.nodes[node]['latent'] += perturb_amount * np.random.randn()
                scm.nodes[node]['latent'] = np.random.randn()
                latent_dict = nx.get_node_attributes(scm, 'latent')
                scm.generate(latent_dict=latent_dict, use_model=True)
                a_max = 0.5
                v_max = 2
                new_a = a_max if scm.nodes['s']['value'] >= (v_max ** 2) / (2 * a_max) else -a_max
                if new_a != state[features.index('a')]:
                    saliency[node] += 1. / saliency_trials

    labels = {node: f'{node}: {saliency[node]:.2f}' for node in scm.nodes if node != 'a'}
    labels['a'] = 'a'
    nx.draw(scm, pos=pos, labels=labels, node_size=2500)
    plt.savefig('./output/obstacle_extended_saliency.png')
    plt.clf()


def test_obstacle_redacted(num_trajectories=100, num_epochs=100):
    # use gaussian noise instead
    def generate_data(num_trajectories):
        # features: x (location), s (measured distance to obstacle), v (velocity)
        # action: a (acceleration)
        # action only consider s
        data = []

        for _ in range(num_trajectories):
            # d = 6 * np.random.randn() + 50
            # d = np.clip(d, 20, None)
            # TODO: Currently using a set d
            d = 50
            a_max = 0.5
            v_max = 2

            x = 0
            s = d - x + np.random.randn()
            v = 0
            a = a_max if s >= (v_max ** 2) / (2 * a_max) else -a_max
            t = np.random.uniform(0.5, 1)

            while x < d:
                data.append((a, s, t, v, x))

                x += v * t + 0.5 * a * (t ** 2)
                v += a * t
                v = min(v, v_max)
                s = d - x + np.random.randn()
                a = a_max if s >= (v_max ** 2) / (2 * a_max) else -a_max

        return np.array(data)

    train_data = generate_data(num_trajectories)
    # only keeps a, s, x
    train_data = np.hstack((train_data[:, :2], train_data[:, -1:]))
    train_data = torch.tensor(train_data).float()
    features = ['a', 's', 'x']
    print(train_data.shape)
    train_data_length = train_data.shape[0]
    train_labels = torch.zeros(train_data_length)
    train_set = [
        (train_data[i], train_labels[i]) for i in range(train_data_length)
    ]

    # create trainable BiCoGAN SCM
    scm = SCM.load_skeleton('skeletons/obstacle_redacted_scm.json', num_latent=1)
    for node in scm.nodes:
        num_parents = len(tuple(scm.predecessors(node)))
        scm.add_model(node, LinearNoiseGenerator(cond_size=num_parents, output_size=1, latent_size=1))

    # train
    losses = []
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        loss = 0
        for n, (samples, _) in enumerate(train_loader):
            loss = scm.optimize(samples)
            losses.append(loss)

        if epoch % 10 == 0:
            print(f'epoch {epoch:03d}: loss = {loss:.4f}')

    scm.save_models('./trained_models/obstacle_redacted_scm.pth')


def visualize_obstacle_redacted():
    def generate_data(num_trajectories):
        # features: x (location), s (measured distance to obstacle), v (velocity)
        # action: a (acceleration)
        # action only consider s
        data = []

        for _ in range(num_trajectories):
            # d = 6 * np.random.randn() + 50
            # d = np.clip(d, 20, None)
            # TODO: Currently using a set d
            d = 50
            a_max = 0.5
            v_max = 2

            x = 0
            s = d - x + np.random.randn()
            v = 0
            a = a_max if s >= (v_max ** 2) / (2 * a_max) else -a_max
            t = np.random.uniform(0.5, 1)

            while x < d:
                data.append((a, s, t, v, x))

                x += v * t + 0.5 * a * (t ** 2)
                v += a * t
                v = min(v, v_max)
                s = d - x + np.random.randn()
                a = a_max if s >= (v_max ** 2) / (2 * a_max) else -a_max

        return np.array(data)

    # create trainable BiCoGAN SCM
    scm = SCM.load_skeleton('skeletons/obstacle_redacted_scm.json', num_latent=1)
    for node in scm.nodes:
        num_parents = len(tuple(scm.predecessors(node)))
        scm.add_model(node, LinearNoiseGenerator(cond_size=num_parents, output_size=1, latent_size=1))
    scm.load_models('./trained_models/obstacle_redacted_scm.pth')

    scm.eval()

    # testing
    np.random.seed(42)

    data = generate_data(1)
    data = np.hstack((data[:, :2], data[:, -1:]))
    features = ['a', 's', 'x']

    plt.plot(data[:, 0] * 10, label='a')
    plt.plot(data[:, 1], label='s')
    plt.plot(data[:, 2], label='x')
    plt.legend()
    plt.savefig('./output/obstacle_state_trend.png')
    plt.clf()


    saliency_trials = 100
    perturb_amount = 1
    saliency = {node: [] for node in scm.nodes if node != 'a'}

    print(data.shape)

    for step in range(data.shape[0]):
        state = data[step]
        for feature, value in zip(features, state):
            scm.add_node(feature, value=value)
        scm.encode()
        latent_dict = nx.get_node_attributes(scm, 'latent')
        latent_dict = {k: float(v) for k, v in latent_dict.items()}
        for node in scm.nodes:
            if node != 'a':
                saliency[node] += [0]
                for _ in range(saliency_trials):
                    perturbed_latent_dict = latent_dict.copy()
                    perturbed_latent_dict[node] += perturb_amount * np.random.randn()
                    scm.generate(latent_dict=perturbed_latent_dict, use_model=True)

                    a_max = 0.5
                    v_max = 2

                    new_a = a_max if scm.nodes['s']['value'] >= (v_max ** 2) / (2 * a_max) else -a_max
                    if new_a != state[features.index('a')]:
                        saliency[node][-1] += 1. / saliency_trials

    for node, ss in saliency.items():
        plt.plot(ss, label=node)
    plt.legend()
    plt.plot()
    plt.savefig('./output/obstacle_redacted_saliency_trend.png')
    # labels = {node: f'{node}: {saliency[node]:.2f}' for node in scm.nodes if node != 'a'}
    # labels['a'] = 'a'
    # nx.draw(scm, pos=pos, labels=labels, node_size=2500)
    # plt.savefig('./output/obstacle_redacted_saliency_trend.png')
    # plt.clf()

    data = generate_data(100)
    data = np.hstack((data[:, :2], data[:, -1:]))

    s = np.linspace(0, 50, 100)[:, np.newaxis]
    a_estimated = scm.nodes['a']['model'].generate(s, z=np.zeros_like(s)).detach().cpu().numpy()
    plt.plot(s, a_estimated, label='estimated')
    plt.scatter(data[:, 1], data[:, 0], label='ground truth', c='orange')
    plt.legend()
    plt.show()
    plt.clf()

    x = np.linspace(0, 50, 100)[:, np.newaxis]
    s_estimated = scm.nodes['s']['model'].generate(x, z=np.zeros_like(x)).detach().cpu().numpy()
    plt.plot(x, s_estimated, label='estimated')
    plt.scatter(data[:, 2], data[:, 1], label='ground truth', c='orange')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # test(300)
    # visualize()
    # test_clinic(1000)
    # visualize_clinic()
    # test_obstacle()
    visualize_obstacle()
    # test_obstacle_linear(100)
    # test_obstacle_redacted(num_epochs=50)
    # visualize_obstacle_redacted()
