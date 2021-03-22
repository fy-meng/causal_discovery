from configparser import ConfigParser
import os
from typing import Dict


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
