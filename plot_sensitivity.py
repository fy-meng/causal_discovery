from collections import defaultdict
import os

import matplotlib.pyplot as plt
import numpy as np

from utils import verify_output_path

plt.rcParams.update({'font.size': 14})

env = 'bang_bang'
test_name = 'perturb'
d = os.path.join('output/sensitivity', env, test_name)

# load data
result = defaultdict(lambda: defaultdict(lambda: []))
for n in os.listdir(d):
    d1 = os.path.join(d, n)
    if os.path.isdir(d1):
        try:
            n = int(n)
        except ValueError:
            try:
                n = float(n)
            except ValueError:
                continue
        for t in os.listdir(d1):
            d2 = os.path.join(d1, t)
            if os.path.isdir(d2):
                saliency_dict = np.load(os.path.join(d2, 'saliency_dict.npz'))
                for k, v in saliency_dict.items():
                    result[k][n].append(v)

# clean up
for k1 in result.keys():
    for k2 in result[k1].keys():
        result[k1][k2] = np.array(result[k1][k2]).T

# plotting
for k in result.keys():
    if '_prev' not in k and k not in ('a', 'left_leg', 'right_leg'):
        plt.figure(dpi=300)
        for j in [0.05, 0.1, 0.15]:
        # for j in sorted(result[k].keys()):
        # for j in [0.05, 0.09, 0.13, 0.17]:
            mean = np.mean(result[k][j], axis=1)
            # mean = np.convolve(mean, np.ones(20), mode='same') / 20
            std = np.std(result[k][j], axis=1)
            plt.plot(mean, label=f'{j}')
        plt.legend(fontsize=16)
        plt.xlabel('Time step', fontsize=20)
        plt.ylabel('Importance', fontsize=20)
        fig_path = os.path.join(d, f'output/{env}_sensitivity_{k}.png')
        verify_output_path(fig_path)
        plt.tight_layout()
        plt.savefig(fig_path)

# plotting norm
norm = defaultdict(lambda: -float('inf'))
for j in sorted(result['s'].keys()):
    for k in result.keys():
        norm[j] = max(norm[j], np.max(result[k][j]))
print(norm)

for k in result.keys():
    if '_prev' not in k and k not in ('a', 'left_leg', 'right_leg'):
        plt.figure(dpi=300)
        for j in [0.05, 0.1, 0.15]:
        # for j in sorted(result[k].keys()):
        # for j in [0.05, 0.09, 0.13, 0.17]:
            mean = np.mean(result[k][j], axis=1)
            # mean = np.convolve(mean, np.ones(20), mode='same') / 20
            mean = mean / norm[j]
            std = np.std(result[k][j], axis=1)
            plt.plot(mean, label=f'{j}')
        plt.legend(fontsize=16)
        plt.xlabel('Time step', fontsize=20)
        plt.ylabel('Normalized importance', fontsize=20)
        fig_path = os.path.join(d, f'output/{env}_sensitivity_{k}_normalized.png')
        verify_output_path(fig_path)
        plt.tight_layout()
        plt.savefig(fig_path)

