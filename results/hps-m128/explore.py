import json
from typing import List, Dict, Any

import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def load_trials(filepath: str) -> List[dict]:
    with open(filepath, 'r') as f:
        trials = json.load(f)
    return trials


def ohe(value: str, choices: List[str]) -> List[int]:
    choice_dict = {choice: i for i, choice in enumerate(choices)}
    x = [0 for _ in range(len(choices))]
    i = choice_dict[value]
    x[i] = 1
    return x


def hps_to_vector(hps: Dict[str, Any]) -> np.ndarray:
    x = [
        hps['latent_size'],
        hps['gaussian_noise'],
        hps['dropout'],
        hps['layer_norm'],
        hps['n_hidden_layers'],
        np.log(hps['learning_rate']),
        hps['max_epochs'],
        hps['batch_size'],
        np.log(hps['weight_decay']),
        hps['early_stopping_patience']
    ]
    x += ohe(hps['activation'], ['relu', 'leakyrelu', 'prelu', 'tanh', 'sigmoid', 'mish', 'selu', 'hardswish'])
    x += ohe(hps['optimizer'], ['adam', 'sgd', 'adamw', 'rmsprop', 'adagrad'])
    x += ohe(hps['sam_type'], ['sam', 'sam-adaptive', 'no-sam'])
    x = np.asarray(x)
    return x


DATASETS = ['ring', 'xor', 'ring+xor', 'ring+xor+sum']
X = []
hps = []
datasets, xs, ys, results = [], [], [], []
best_results = []
maxima = []
for dataset in DATASETS:
    dataset_results = []
    row = []
    for trial in load_trials(f'hps-{dataset}.json'):
        results.append((trial['auroc'] + trial['auprc'], trial['auroc'], trial['auprc'], trial['params']))
        dataset_results.append(results[-1])
        ys.append(trial['auroc'])
        xs.append(trial['params']['latent_size'])
        row.append(ys[-1])
        X.append(hps_to_vector(trial['params']))
        datasets.append(dataset)
    results = list(sorted(results))
    maxima.append(np.max(row))
    best_results.append(list(sorted(dataset_results))[-1])
    print(best_results[-1])
xs = np.asarray(xs)
ys = np.asarray(ys)
X = np.asarray(X)
datasets = np.asarray(datasets, dtype=object)

D = scipy.spatial.distance.cdist(X, X, metric='mahalanobis')
coords = TSNE(metric='precomputed', init='random').fit_transform(D)

f, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [10, 12]}, figsize=(15, 6))

ax2 = plt.subplot(1, 2, 2)
sc = ax2.scatter(coords[:, 0], coords[:, 1], c=ys, alpha=0.7)
ax2.grid(linestyle='--', alpha=0.4, color='grey', linewidth=0.5)
ax2.spines[['right', 'top']].set_visible(False)
plt.colorbar(sc)
ax2.set_title('AUROC of each trial')
ax2.set_xlabel('First component')
ax2.set_ylabel('Second component')

colors = ['slateblue', 'teal', 'firebrick', 'goldenrod']
ax = plt.subplot(1, 2, 1)
for k, dataset in enumerate(DATASETS):
    mask = (datasets == dataset)
    print(coords[mask, 0].shape, coords[mask, 1].shape)
    ax.scatter(coords[mask, 0], coords[mask, 1], alpha=0.7, label=dataset.upper(), color=colors[k])
    mask = np.logical_and(mask, ys == maxima[k])
    ax2.scatter(coords[mask, 0], coords[mask, 1], alpha=0.8, color=colors[k], marker='x', s=300)
ax.grid(linestyle='--', alpha=0.4, color='grey', linewidth=0.5)
ax.spines[['right', 'top']].set_visible(False)
ax.set_title('HP optimization trials of each dataset')
ax.set_xlabel('First component')
ax.set_ylabel('Second component')
ax.legend()

plt.tight_layout()
plt.savefig('hps-m128-tsne.png', dpi=200)

plt.show()
