from configparser import ConfigParser
import os
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim


def verify_output_path(output_path):
    output_dir = os.path.split(output_path)[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def load_config() -> Dict[str, object]:
    parser = ConfigParser()
    parser.read('./config.ini')
    config = {}
    for section in parser.sections():
        for key, item in parser[section].items():
            # convert to list of int
            if key in ('hidden_layers', 'action_min', 'action_max'):
                config[key] = [int(s) for s in item.split(',')]
                continue
            # try convert to int
            try:
                config[key] = int(item)
                continue
            except ValueError:
                pass
            # convert to float
            try:
                config[key] = float(item)
                continue
            except ValueError:
                pass
            # convert to bool
            if item == 'True' or item == 'False':
                config[key] = bool(item)
                continue
            # check for empty value
            if item == '':
                config[key] = None
                continue
            # otherwise, kept as str
            else:
                config[key] = item
    return config


# noinspection PyAbstractClass
class FCLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=True):
        super(FCLayer, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.activation = nn.ReLU(inplace=True)
        # self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.5) if dropout else None

    def forward(self, x):
        output = self.fc(x)
        output = self.activation(output)
        if self.dropout:
            output = self.dropout(output)
        return output


# noinspection PyAbstractClass
class FCNet(nn.Module):
    """
    A fully-connected neural net in which each layer is consist of Linear ->
    Activation -> Dropout, and an additional Linear layer at the end.
    """

    def __init__(self, in_channels, out_channels, hidden_layers, dropout=True, sigmoid=False):
        super(FCNet, self).__init__()
        layers = []
        prev_channels = in_channels
        for channels in hidden_layers:
            layers.append(FCLayer(prev_channels, channels, dropout=dropout))
            prev_channels = channels
        layers.append(nn.Linear(prev_channels, out_channels))
        if sigmoid:
            layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Net:
    def __init__(self, in_channels, out_channels, layers, dropout=False, sigmoid=False, lr=1e-3):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model = FCNet(in_channels, out_channels, layers, dropout=dropout, sigmoid=sigmoid)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr)

    def predict(self, *input_vars):
        input_vars = list(input_vars)
        for i in range(len(input_vars)):
            if len(input_vars[i].shape) == 1:
                input_vars[i] = input_vars[i].unsqueeze(-1)

        x = torch.cat(input_vars, dim=1)
        return self.model(x)

    def save_model(self, model_save_path):
        verify_output_path(model_save_path)
        torch.save(self.model.state_dict(), model_save_path)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
