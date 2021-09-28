from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import pylab
from scipy.special import softmax
import scipy

plt.rcParams.update({'font.size': 14})


def lunar_lander():
    features = {
        'x_pos': 'X position',
        'y_pos': 'Y position',
        'x_vel': 'X velocity',
        'y_vel': 'Y velocity',
        'angle': 'Angle',
        'angle_vel': 'Angular velocity',
        'left_leg': 'Left leg',
        'right_leg': 'Right leg'
    }

    # lunar lander: legend
    figlegend = pylab.figure(figsize=(10, 2), dpi=300)
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    for f in features.keys():
        ax.plot(np.ones(10), label=features[f])
    figlegend.legend(ax.lines, features.values(), 'center', ncol=4)
    figlegend.savefig('lunar_lander_legend.png')
    plt.clf()

    # lunar lander: trajectory
    trajectory = pd.read_pickle('./causal_data/lunar_lander/trajectory.pkl')

    plt.figure(dpi=300)
    for f in features.keys():
        plt.plot(trajectory[f], label=features[f])
    plt.xlabel('Time step')
    plt.ylabel('Value')
    plt.savefig('lunar_lander_trajectory.png', bbox_inches='tight')
    plt.clf()

    # lunar lander: curr step
    s_our: pd.DataFrame = pd.read_pickle('./causal_data/lunar_lander/saliency_smoothed.pkl')
    s_ext: pd.DataFrame = pd.read_pickle('./causal_data/lunar_lander/saliency_existing_smoothed.pkl')
    s_ext.rename(columns={'ang_vel': 'angle_vel'}, inplace=True)

    print(s_our.keys())
    print(s_ext.keys())

    plt.figure(dpi=300)
    cmap = plt.get_cmap("tab10")
    for f in features.keys():
        plt.plot(s_our[f], label=features[f])
    plt.xlabel('Time step', fontsize=16)
    plt.ylabel('Prev-step importance', fontsize=16)
    plt.savefig('lunar_lander_ours_curr.png', bbox_inches='tight')
    plt.clf()

    # lunar lander: diff
    plt.figure(dpi=300)
    cmap = plt.get_cmap("tab10")
    for f in ['x_pos', 'y_pos']:
        plt.plot(s_our[f] - s_ext[f], label=features[f])
    plt.plot(s_our['angle'] - s_ext['angle'], label=features['angle'], color=cmap(4))
    plt.plot(np.zeros_like(s_our['x_pos']), label='X vel, Y vel, Angle vel,\nL leg, R leg', color=cmap(2))
    plt.legend(fontsize=14)
    plt.xlabel('Time step', fontsize=16)
    plt.ylabel('Prev-step importance', fontsize=16)
    plt.savefig('lunar_lander_diff.png', bbox_inches='tight')
    plt.clf()

    # lunar lander: prev step
    plt.figure(dpi=300)
    cmap = plt.get_cmap("tab10")
    for f in features.keys():
        plt.plot(s_our[f + '_prev'], label=features[f])
    plt.xlabel('Time step', fontsize=16)
    plt.ylabel('Prev-step importance', fontsize=16)
    plt.savefig('lunar_lander_ours_prev.png', bbox_inches='tight')
    plt.clf()


def bang_bang():
    plt.rcParams.update({'font.size': 16})
    features = {
        'v': 'Velocity V_t',
        'x': 'Position X_t',
        's': 'Distance D_t',
        'a': 'Acceleration A_t'
    }

    trajectory: pd.DataFrame = pd.read_pickle('./causal_data/bang_bang_control/trajectory.pkl')
    saliency: pd.DataFrame = pd.read_pickle('./causal_data/bang_bang_control/saliency.pkl')

    # bang bang: trajectory
    plt.figure(dpi=300)
    for f in features.keys():
        plt.plot(trajectory[f], label=features[f])
    plt.xlabel('Time step', fontsize=20)
    plt.ylabel('Value', fontsize=20)
    plt.legend(fontsize=16, loc='right')
    plt.savefig('bangbang_trajectory.png', bbox_inches='tight')
    plt.clf()

    # bang bang: trajectory
    plt.figure(dpi=300)
    plt.plot(saliency['x'], label='X_t, D_t')
    plt.plot(saliency['x_prev'], label='X_{t-1}')
    plt.plot(saliency['v'], label='V_t, V_{t-1}\nD_{t-1}, A_{t-1}')
    plt.xlabel('Time step', fontsize=20)
    plt.ylabel('Importance', fontsize=20)
    plt.legend(fontsize=16, loc='upper left')
    plt.savefig('bangbang_saliency.png', bbox_inches='tight')
    plt.clf()


# lunar_lander()
bang_bang()
