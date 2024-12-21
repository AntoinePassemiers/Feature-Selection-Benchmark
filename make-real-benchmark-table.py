import os
import json
import math

import numpy as np


ROOT = os.path.dirname(os.path.abspath(__file__))


METHOD_NAMES = [
    'cae',
    'fsnet',
    'mrmr',
    'treeshap',
    'deeppink',
    'canceloutsoftmax',
    'Deconvolution',
    'GuidedBackprop',
    'FeaturePermutation',
    'Saliency',
    'DeepLift',
    'IG_noMul',
    'SmoothGrad',
    'mi',
    'relief',
    'ShapleyValueSampling',
    'canceloutsigmoid',
    'FeatureAblation',
    'InputXGradient',
    'lassonet',
    'rf',
]
#DATASETS = ['fashion', 'isolet', 'mice', 'har', 'mnist', 'coil20']
DATASETS = ['arcene', 'madelon', 'dexter', 'dorothea', 'gisette'] + ['fashion', 'isolet', 'mice', 'har', 'mnist', 'coil20']


def compute_ranks(scores: np.ndarray) -> np.ndarray:
    ranks = np.copy(scores)
    for i in range(len(scores)):
        if np.isnan(scores[i]):
            ranks[i] = np.sum(~np.isnan(scores)) + 1
        else:
            ranks[i] = np.sum(scores > scores[i]) + 1
    return ranks


results = np.full((len(DATASETS), len(METHOD_NAMES)), np.nan, dtype=[('auroc', 'f4'), ('aupr', 'f4'), ('time', 'f4'), ('peak_mem', 'f4')])
completion_rate = []
for i, dataset in enumerate(DATASETS):
    for j, method in enumerate(METHOD_NAMES):
        filepath = os.path.join(ROOT, 'results', 'external-data', f'{dataset}-{method}.json')
        completion_rate.append(os.path.exists(filepath))
        if not os.path.exists(filepath):
            print(filepath)
        else:
            with open(filepath, 'r') as f:
                data = json.load(f)
            results[i, j] = (data['auroc'], data['aupr'], data['computation-time'], data['max-memory-usage'])
results = results.T
print(f'Completion rate: {np.mean(completion_rate)}')

scores = np.sqrt(results['aupr'] * results['auroc'])
#scores = np.nan_to_num(np.sqrt(results['aupr'] * results['auroc']))
overall_scores = np.copy(scores)
for k in range(overall_scores.shape[1]):
    overall_scores[:, k] = compute_ranks(overall_scores[:, k])
overall_scores = np.mean(overall_scores, axis=1)

peak_mem = 1e-6 * np.nanmean(results['peak_mem'], axis=1)
table = np.concatenate((
    np.nanmean(results['time'], axis=1)[:, np.newaxis],
    peak_mem[:, np.newaxis],
    scores,
    overall_scores[:, np.newaxis],
), axis=1)
col_names = ['Time (s)', 'Peak mem'] + DATASETS + ['Average rank']
row_names = METHOD_NAMES

print(f'Method & ' + ' & '.join(col_names))
for i in range(len(row_names)):
    row = []
    for j in range(2, len(col_names)):
        if np.isnan(table[i, j]):
            row.append('  -  ')
        else:
            s = f'{table[i, j]:.3f}'
            row.append(s)
    s = '{:<20}'.format(METHOD_NAMES[i]) + ' & '
    if not np.isnan(table[i, 0]):
        s += '{:<5}'.format(int(math.ceil(table[i, 0]))) + ' & '
        s += '{:<5}'.format(int(math.ceil(table[i, 1]))) + ' & '
    else:
        s += '{:<5}'.format(' ') + ' & '
        s += '{:<6}'.format(' ') + ' & '
    s += ' & '.join(row)
    s += ' \\\\'
    print(s)
