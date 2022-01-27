from collections import defaultdict

import numpy as np
import matplotlib.patches
import matplotlib.pyplot as plt
import pandas as pd
import pylab

plt.rcParams.update({'font.size': 14})


def lunar_lander():
    features = {
        'x_pos': 'X pos.',
        'y_pos': 'Y pos.',
        'x_vel': 'X vel.',
        'y_vel': 'Y vel.',
        'angle': 'Angle',
        'angle_vel': 'Ang. vel.',
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
    s_ext: pd.DataFrame = pd.read_pickle('./causal_data/lunar_lander/saliency_existing_smoothed2.pkl')
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
    plt.plot(np.zeros_like(s_our['x_pos']), label='X vel., Y vel., Ang. vel.,\nL leg, R leg', color=cmap(2))
    plt.legend(fontsize=14)
    plt.xlabel('Time step', fontsize=16)
    plt.ylabel('Importance difference', fontsize=16)
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


def lunar_lander_new():
    features = {
        'x_pos': 'X pos.',
        'y_pos': 'Y pos.',
        'x_vel': 'X vel.',
        'y_vel': 'Y vel.',
        'angle': 'Angle',
        'angle_vel': 'Ang. vel.',
        'left_leg': 'Left leg',
        'right_leg': 'Right leg'
    }

    # lunar lander: trajectory
    # trajectory = pd.read_pickle('./causal_data/lunar_lander/trajectory.pkl')
    #
    # plt.figure(dpi=300)
    # for f in features.keys():
    #     plt.plot(trajectory[f], label=features[f])
    # plt.xlabel('Time step')
    # plt.ylabel('Value')
    # plt.show()

    bar_size = 0.3

    s_our = dict(np.load('output/lunar_lander_new/saliency_dict.npz'))
    s_ext = dict(np.load('output/lunar_lander_saliency_map/saliency_dict.npz'))

    for k in s_our.keys():
        s_our[k] = np.convolve(s_our[k], np.ones(20), mode='same') / 20
        s_ext[k] = np.convolve(s_our[k], np.ones(20), mode='same') / 20

    # lunar lander: diff
    cmap = plt.get_cmap("tab10")
    plt.figure(dpi=300)
    for f in ['x_pos', 'y_pos']:
        plt.plot(s_our[f] - s_ext[f], label=features[f])
    plt.plot(s_our['angle'] - s_ext['angle'], label=features['angle'], color=cmap(4))
    plt.plot(np.zeros_like(s_our['x_pos']), label='X vel., Y vel., Ang. vel.,\nL leg, R leg', color=cmap(2))
    plt.xlabel('Time step')
    plt.ylabel('Importance difference')
    plt.legend()
    plt.tight_layout()
    plt.savefig('lunar_lander_diff.png')
    plt.clf()

    plt.figure(dpi=300)
    for f in features.keys():
        plt.plot(s_our[f], label=features[f])
    # plt.legend()
    plt.ylim(0, 3.5)
    plt.xlabel('Time step')
    plt.ylabel('Causal importance')
    plt.tight_layout()
    plt.savefig('lunar_lander_curr.png')
    plt.clf()

    plt.figure(dpi=300)
    for f in features.keys():
        plt.plot(s_our[f + '_prev'], label=features[f])
    # plt.legend()
    plt.ylim(0, 3.5)
    plt.xlabel('Time step')
    plt.ylabel('Prev-step causal importance')
    plt.tight_layout()
    plt.savefig('lunar_lander_prev.png')
    plt.clf()

    t0 = np.argmax(s_our['angle_vel'][:70])
    t1 = np.argmax(s_our['x_vel'][100:150]) + 100
    t2 = np.argmax(s_our['x_pos'][200:]) + 200
    print(t0, t1, t2)
    for t in [t0, t1, t2]:
        fig, ax = plt.subplots(dpi=300)
        ax: plt.Axes

        cmap = plt.get_cmap("tab10")

        xs_prev = np.arange(1, len(features) + 1)
        xs_curr = xs_prev - bar_size

        bars_curr = ax.bar(xs_curr, [s_our[f][t] for f in features.keys()], label='Curr. step',
                           align='edge', width=bar_size)
        bars_prev = ax.bar(xs_prev, [s_our[f + '_prev'][t] for f in features.keys()], label='Prev. step',
                           align='edge', width=bar_size, fill=False, hatch=r'///')
        for i in range(len(features)):
            bars_curr[i].set_color(cmap(i))
            bars_prev[i].set_color(cmap(i))

        p1 = matplotlib.patches.Patch(facecolor='000000', fill=True, label='Curr. step')
        p2 = matplotlib.patches.Patch(facecolor='000000', fill=False, hatch=r'///', label='Prev. step')
        ax.legend(handles=[p1, p2], fontsize=14)
        ax.set_ylim(0, 3.5)
        # ax.set_yticks([0, 0.5, 1, 1.5, 2])
        ax.set_xticks(range(1, len(features) + 1))
        ax.set_xticklabels(features.values(), rotation=-30, ha='left', rotation_mode='anchor')
        ax.set_ylabel('Causal Importance')
        plt.tight_layout()
        plt.savefig(f'lunar_lander_{t}.png')
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


def crop2():
    plt.rcParams.update({'font.size': 16})
    features = {
        'precip': 'Precipitation',
        'humidity': 'Humidity',
        'weight': 'Crop weight',
        'radiation': 'Radiation',
        'a': 'Irrigation'
    }

    trajectory = np.load('./output/crop2_q/trajectory.npy').T
    trajectory = {
        'humidity': trajectory[2],
        'precip': trajectory[4],
        'radiation': trajectory[6],
        'weight': trajectory[8],
        'a': trajectory[0]
    }
    trajectory = pd.DataFrame(trajectory)
    importance = np.load('./output/crop2_q/saliency_dict.npz')
    saliency = np.load('./output/crop2_saliency_map/saliency_dict.npz')

    # crop2: trajectory
    plt.figure(dpi=300)
    for f in features.keys():
        plt.plot(trajectory[f], label=features[f])
    plt.xlabel('Time step', fontsize=20)
    plt.ylabel('Value', fontsize=20)
    plt.ylim(-0.2, 1.5)
    plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
    plt.xticks([0, 2, 4, 6, 8, 10])
    plt.legend(fontsize=16, loc='upper left')
    plt.savefig('crop2_trajectory.png', bbox_inches='tight')
    plt.clf()

    # crop2: importance
    plt.figure(dpi=300)
    plt.plot(importance['precip'], label='Precipitation')
    plt.plot(importance['humidity'], label='Humidity')
    plt.plot(importance['weight'], label='Crop weight')
    plt.plot(importance['radiation'], label='Radiation')
    plt.xlabel('Time step', fontsize=20)
    plt.ylabel('Importance', fontsize=20)
    plt.ylim(-0.2, 3.2)
    plt.xticks([0, 2, 4, 6, 8, 10])
    plt.legend(fontsize=16, loc='upper left')
    plt.savefig('crop2_importance.png', bbox_inches='tight')
    plt.clf()

    # crop2: saliency
    plt.figure(dpi=300)
    plt.plot(saliency['precip'], label='Precipitation')
    plt.plot(saliency['humidity'], label='Humidity')
    plt.plot(saliency['weight'], label='Crop weight')
    plt.plot(saliency['radiation'], label='Radiation')
    plt.xlabel('Time step', fontsize=20)
    plt.ylabel('Importance', fontsize=20)
    plt.ylim(-0.2, 3.2)
    plt.xticks([0, 2, 4, 6, 8, 10])
    plt.legend(fontsize=16, loc='upper left')
    plt.savefig('crop2_saliency.png', bbox_inches='tight')
    plt.clf()

    # crop2: single point
    bar_size = 0.3
    t = 5
    print('state:')
    for f in features:
        print(f, trajectory[f][t])
    fig, ax = plt.subplots(dpi=300)
    ax: plt.Axes
    cmap = plt.get_cmap("tab10")
    xs_ext = np.arange(1, len(features))
    xs_our = xs_ext - bar_size
    bars_our = ax.bar(xs_our, [importance[f][t] for f in list(features.keys())[:-1]], label='Ours',
                      align='edge', width=bar_size)
    bars_ext = ax.bar(xs_ext, [saliency[f][t] for f in list(features.keys())[:-1]], label='Saliency map',
                      align='edge', width=bar_size, fill=False, hatch=r'///')
    for i in range(len(features) - 1):
        bars_our[i].set_color(cmap(i))
        bars_ext[i].set_color(cmap(i))

    p1 = matplotlib.patches.Patch(facecolor='000000', fill=True, label='Ours')
    p2 = matplotlib.patches.Patch(facecolor='000000', fill=False, hatch=r'///', label='Saliency map')
    ax.legend(handles=[p1, p2], fontsize=14)
    ax.set_ylim(0, 2.5)
    # ax.set_yticks([0, 0.5, 1, 1.5, 2])
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels([features[f] for f in list(features.keys())[:-1]], rotation=-15)
    # labels = ax.set_xticklabels([features[f] for f in list(features.keys())[:-1]])
    # for i, label in enumerate(labels):
    #     label.set_y(label.get_position()[1] - (i % 2) * 0.075)
    ax.set_ylabel('Importance')
    plt.tight_layout()
    plt.savefig('crop2_single_point.png', bbox_inches='tight')
    plt.clf()


# lunar_lander()
lunar_lander_new()
# bang_bang()
# crop2()
