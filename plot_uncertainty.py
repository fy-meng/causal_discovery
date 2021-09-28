import os

import numpy as np
import matplotlib.pyplot as plt

d = './output/blackjack_uncertainty'
num_tries = 20

plt.figure(dpi=300)

trajectory = np.load(d + f'/0/trajectory.npy')
for i in range(trajectory.shape[1]):
    plt.plot(trajectory[:, i])
plt.savefig('trajectory.png')
plt.clf()

result = None
for t in range(num_tries):
    saliency = np.load(d + f'/{t}/saliency_dict.npz')
    for k in saliency.keys():
        if result is None:
            result = {_k: np.zeros((num_tries, len(saliency[k]))) for _k in saliency.keys()}
        result[k][t] = saliency[k]

saliency_truth = np.load('./output/blackjack/saliency_dict.npz')

for k in result.keys():
    # plot all
    # for i in range(result[k].shape[0]):
    #     plt.plot(result[k][i])

    # plot min max
    # plt.plot(np.min(result[k], axis=0))
    # plt.plot(np.max(result[k], axis=0))

    # plot error bar
    plt.errorbar(np.arange(result[k].shape[1]),
                 np.mean(result[k], axis=0),
                 yerr=np.vstack((np.mean(result[k], axis=0) - np.min(result[k], axis=0),
                                 np.max(result[k], axis=0) - np.mean(result[k], axis=0))),
                 label='bootstrap')
    plt.plot(saliency_truth[k], label='truth')
    plt.legend()
    plt.title(k)
    plt.savefig(f'{k}.png')
    plt.clf()
