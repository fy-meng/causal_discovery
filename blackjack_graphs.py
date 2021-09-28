import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pylab

plt.rcParams.update({'font.size': 14})

saliency_q = dict(np.load('./output/blackjack/saliency_dict.npz', allow_pickle=True))
saliency_a = dict(np.load('./output/blackjack_a/saliency_dict.npz', allow_pickle=True))
length = len(saliency_q['hand'])
features = ['hand', 'hand_prev', 'ace', 'ace_prev', 'dealer', 'dealer_prev', 'a_prev']
feature_dict = {
    'hand': 'Hand',
    'hand_prev': 'Hand (Prev)',
    'ace': 'Ace',
    'ace_prev': 'Ace (Prev)',
    'dealer': 'Dealer',
    'dealer_prev': 'Dealer (Prev)',
    'a_prev': 'Action (Prev)'
}
cm = plt.rcParams['axes.prop_cycle'].by_key()['color']

for f in features:
    if f.endswith('_prev'):
        saliency_q[f][0] = 0
        saliency_a[f][0] = 0

bins = [0]
for _ in range(7):
    bins += [1, 2, 2]
bins += [1]
bins = np.cumsum(bins)

max_val = max([max(saliency_q[f]) for f in features])

for i in range(length):
    fig, ax = plt.subplots(dpi=300)
    ax: plt.Axes

    weights = np.zeros(len(bins) - 1)
    for j in range(7):
        weights[3 * j + 1] = saliency_q[features[j]][i]
        weights[3 * j + 2] = saliency_a[features[j]][i] * max_val

    n, bins, patches = ax.hist(bins[:-1], bins=bins, weights=weights)
    ax.set_xticks(np.arange(3, 34, 5))
    ax.set_xticklabels(['' for _ in range(7)])
    ax.set_ylim(0, 0.175)
    # ax.set_yticks([0, 0.5, 1])
    plt.tight_layout()

    for j in range(7):
        plt.setp(patches[3 * j + 1], 'fc', cm[j])
        plt.setp(patches[3 * j + 2], 'ec', cm[j])
        plt.setp(patches[3 * j + 2], 'fill', False)
        plt.setp(patches[3 * j + 2], 'hatch', '///')

    p1 = mpatches.Patch(facecolor='000000', fill=True, label='Q-value based')
    p2 = mpatches.Patch(facecolor='000000', fill=False, hatch=r'///', label='Action based')
    ax.legend(handles=[p1, p2], loc=2, fontsize=14)
    ax.set_ylabel('Q-value based importance', fontsize=14)

    ax2: plt.Axes = ax.twinx()
    ax2.set_ylabel('Action based importance', fontsize=14)
    ax2.set_ylim(0, 0.175)
    ax2.set_yticks([0, 0.5 * max_val, max_val])
    ax2.set_yticklabels(['0.0', '0.5', '1.0'])

    plt.tight_layout()

    plt.savefig(f'./{i}.png', bbox_inches='tight')


# legend
figlegend = pylab.figure(figsize=(10, 2), dpi=300)
fig = plt.figure(dpi=300)
ax = fig.add_subplot(111)
ax.hist(np.ones(4))
patches = ax.patches
for i in range(7):
    plt.setp(patches[i], 'fc', cm[i])
    # plt.setp(patches[3 * j + 2], 'ec', cm[i])
    # plt.setp(patches[3 * j + 2], 'fill', False)
    # plt.setp(patches[3 * j + 2], 'hatch', '///')
figlegend.legend(ax.patches, feature_dict.values(), 'center', ncol=2, fontsize=16)
figlegend.savefig('legend2.png')
figlegend.clf()

p1 = mpatches.Patch(facecolor='000000', fill=True, label='Q-value based')
p2 = mpatches.Patch(facecolor='000000', fill=False, hatch=r'///', label='Action based')
figlegend.legend(handles=[p1, p2], loc='center', ncol=2, fontsize=16)
figlegend.savefig('legend1.png')

