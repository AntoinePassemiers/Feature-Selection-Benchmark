# -*- coding: utf-8 -*-
#
#  make-figures.py
#
#  Copyright 2022 Antoine Passemiers <antoine.passemiers@gmail.com>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, 'data')
RESULTS_PATH = os.path.join(ROOT, 'results')
OUTPUT_PATH = os.path.join(ROOT, 'figures')
if not os.path.isdir(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
TABLES_PATH = os.path.join(ROOT, 'tables')
if not os.path.isdir(TABLES_PATH):
    os.makedirs(TABLES_PATH)

METHODS = {
    'NN': ('Neural network (no FS)', 'grey', 'x'),
    'Saliency': ('Saliency maps', 'black', 'x'),
    'InputXGradient': (r'Input $\times$ Gradient', 'gold', 'o'),
    'IG_noMul': ('Integrated gradient', 'olive', 'o'),
    'SmoothGrad': ('SmoothGrad', 'brown', 'x'),
    'GuidedBackprop': ('Guided backpropagation', 'darkcyan', 'x'),
    'DeepLift': ('DeepLift', 'orangered', 'D'),
    'Deconvolution': ('Deconvolution', 'mediumseagreen', 'x'),
    'FeatureAblation': ('Feature ablation', 'goldenrod', 'D'),
    'FeaturePermutation': ('Feature permutation', 'blue', '*'),
    'ShapleyValueSampling': ('Shapley value sampling', 'mediumpurple', 'D'),
    'CancelOut_Sigmoid': ('CancelOut (sigmoid)', 'orangered', 'o'),
    'CancelOut_Softmax': ('CancelOut (softmax)', 'orange', 'o'),
    'DeepPINK_U': ('DeepPINK (U)', 'violet', 'x'),
    'DeepPINK_2o': ('DeepPINK (2o)', 'pink', 'x'),
    #'DeepPINK_DK': ('DeepPINK (DK)', 'violet', 'x'),
    'CAE': ('Concrete Autoencoder', 'mediumseagreen', 'D'),
    'FSNet': ('FSNet', 'navy', 's'),
    'RF': ('Random Forest', 'darkgreen', 'D'),
    'Relief': ('Relief', 'turquoise', 'o'),
    'mRMR': ('mRMR', 'steelblue', 's'),
    'LassoNet': ('LassoNet', 'chocolate', '*')
}
METHOD_NAMES = list(METHODS.keys())


def load_extra_results(filepath, results):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    header = lines[0].rstrip().split('\t')
    lines = lines[1:]
    for line in lines:
        elements = line.rstrip().split('\t')
        if len(elements) > 1:
            n_features = int(elements[0].replace('feat.csv', '').split('-')[-1])
            for i in range(1, len(elements)):
                if header[i].endswith('_AUC'):
                    results[header[i].replace('_AUC', '')][n_features]['auroc'] = float(elements[i])
                elif header[i].endswith('_AUPRC'):
                    results[header[i].replace('_AUPRC', '')][n_features]['auprc'] = float(elements[i])
                elif header[i].endswith('_bestK'):
                    results[header[i].replace('_bestK', '')][n_features]['best-k'] = float(elements[i])
                elif header[i].endswith('_best2K'):
                    results[header[i].replace('_best2K', '')][n_features]['best-2k'] = float(elements[i])
                else:
                    raise NotImplementedError(header[i])


def load_nn_results(filepath, results):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    header = lines[0].rstrip().split('\t')
    lines = lines[1:]
    for line in lines:
        elements = line.rstrip().split('\t')
        if len(elements) > 1:
            n_features = int(elements[0].replace('feat.csv', '').split('-')[-1])
            results['NN'][n_features]['auroc'] = float(elements[1])
            results['NN'][n_features]['auprc'] = float(elements[2])
            for i in range(3, len(elements)):
                if header[i].endswith('_bestK'):
                    results[header[i].replace('_bestK', '')][n_features]['best-k'] = float(elements[i])
                elif header[i].endswith('_best2K'):
                    results[header[i].replace('_best2K', '')][n_features]['best-2k'] = float(elements[i])
                else:
                    raise NotImplementedError(header[i])


def plot_performance(results, ns, k):

    ns = np.asarray(ns)

    alpha = 0.5
    legend_size = 9

    plt.figure(figsize=(16, 8))

    ax = plt.subplot(2, 2, 1)
    for method_name in METHOD_NAMES:
        label, color, marker = METHODS[method_name]
        ys = np.asarray([results[method_name][n_features]['auroc'] for n_features in ns])
        if not np.all(np.isnan(ys)):
            plt.plot(ns, 100 * ys, alpha=alpha, label=label, color=color, marker=marker)
    plt.xscale('log')
    plt.ylabel('AUROC (%)', fontname='Century gothic')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n_features) for n_features in ns])

    ax = plt.subplot(2, 2, 2)
    for method_name in METHOD_NAMES:
        label, color, marker = METHODS[method_name]
        ys = np.asarray([results[method_name][n_features]['auprc'] for n_features in ns])
        if not np.all(np.isnan(ys)):
            plt.plot(ns, 100 * ys, alpha=alpha, label=label, color=color, marker=marker)
    plt.xscale('log')
    plt.ylabel('AUPRC (%)', fontname='Century gothic')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n_features) for n_features in ns])
    prop = {'family': 'Century gothic', 'size': legend_size}
    plt.legend(prop=prop, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, frameon=False)

    ax = plt.subplot(2, 2, 3)
    for method_name in METHOD_NAMES:
        label, color, marker = METHODS[method_name]
        ys = np.asarray([results[method_name][n_features]['best-k'] for n_features in ns])
        if not np.all(np.isnan(ys)):
            plt.plot(ns, 100 * ys, alpha=alpha, label=label, color=color, marker=marker)
    plt.fill_between(ns, 100 * k / ns, color='grey', alpha=0.25)
    plt.xscale('log')
    plt.xlabel('Number of features', fontname='Century gothic')
    plt.ylabel('Best p (%)', fontname='Century gothic')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n_features) for n_features in ns])

    ax = plt.subplot(2, 2, 4)
    for method_name in METHOD_NAMES:
        label, color, marker = METHODS[method_name]
        ys = np.asarray([results[method_name][n_features]['best-2k'] for n_features in ns])
        if not np.all(np.isnan(ys)):
            plt.plot(ns, 100 * ys, alpha=alpha, label=label, color=color, marker=marker)
    plt.fill_between(ns, np.minimum(100, 2 * 100 * k / ns), color='grey', alpha=0.25)
    plt.xscale('log')
    plt.xlabel('Number of features', fontname='Century gothic')
    plt.ylabel('Best 2p (%)', fontname='Century gothic')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n_features) for n_features in ns])
    prop = {'family': 'Century gothic', 'size': legend_size}
    plt.legend(prop=prop, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, frameon=False)

    plt.tight_layout()


def create_table(results):
    s = '\\begin{table}[h!]\n'
    s += '\\centering\n'
    s += '\\caption{Average of best k and best 2k score percentages across all values of $p \\in \\{2, 4, 6, 8, 16, 32, 64, 128, 256, 512\\}$, on the 4 datasets. Top and bottom parts of the table correspond to instance-level feature attribution and embedded/filter FS methods, respectively. Best performing methods are highlighted in bold.}\n'
    s += '\\begin{adjustwidth}{-1.0in}{-1.0in}\n'
    s += '\\label{tab:benchmark}{\\begin{tabular}{lrrrrrrrr}\\toprule\n'
    s += 'Dataset & \\multicolumn{2}{c}{\\ring} & \\multicolumn{2}{c}{\\xor} & \\multicolumn{2}{c}{\\ringxor} & \\multicolumn{2}{c}{\\ringxorsum} \\\\\n'
    s += '\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-9}\n'
    s += 'Method & Best p & Best 2p & Best p & Best 2p & Best p & Best 2p & Best p & Best 2p \\\\ \\midrule\n'

    def f(method_name):
        s = METHODS[method_name][0]
        for dataset_name in ['ring', 'xor', 'ring+xor', 'ring+xor+sum']:
            best_k = 100 * np.nanmean([results[dataset_name][method_name][n_features]['best-k'] for n_features in results[dataset_name][method_name].keys()])
            best_2k = 100 * np.nanmean([results[dataset_name][method_name][n_features]['best-2k'] for n_features in results[dataset_name][method_name].keys()])
            s += f' & {best_k:.1f} & {best_2k:.1f}'
        s += ' \\\\\n'
        return s

    for method_name in ['Saliency', 'IG_noMul', 'DeepLift', 'InputXGradient', 'SmoothGrad', 'GuidedBackprop',
                        'Deconvolution', 'FeatureAblation', 'FeaturePermutation', 'ShapleyValueSampling']:
        s += f(method_name)
    s += '\\midrule\n'
    for method_name in ['mRMR', 'LassoNet', 'Relief', 'CAE', 'FSNet', 'CancelOut_Softmax', 'CancelOut_Sigmoid',
                        'DeepPINK_U', 'DeepPINK_2o', 'RF']:
        s += f(method_name)

    s += '\\bottomrule\n'
    s += '\\end{tabular}}{}\n'
    s += '\\end{adjustwidth}\n'
    s += '\\end{table}\n'
    return s


# Plot datasets
def load_dataset(filename, n_features):
    filepath = os.path.join(DATA_PATH, filename)
    column_names = [f'x{i}' for i in range(n_features)] + ['y']
    df = pd.read_csv(filepath, delimiter=',', header=None, names=column_names)
    y = df['y'].to_numpy().astype(int)
    df.drop('y', axis=1, inplace=True)
    X = df.to_numpy().astype(np.float32)
    return X, y
'ring-xor-sum_1000samples-6feat.csv'
'ring+xor-4feat.csv'
'ring-2feat.csv'

datasets = [
    ('ring_1000samples-2feat.csv', 2, 'RING'),
    ('xor_1000samples-2feat.csv', 2, 'XOR'),
    ('ring+xor_1000samples-4feat.csv', 4, 'RING+XOR'),
    ('ring-xor-sum_1000samples-6feat.csv', 6, 'RING+XOR+SUM'),
]
_, axs = plt.subplots(3, 4, figsize=(16, 8))
for j, (filename, n_features, title) in enumerate(datasets):
    X, y = load_dataset(filename, n_features)
    for i in range(0, n_features // 2):
        if i == 0:
            axs[i, j].set_title(title, fontname='Century gothic', fontdict={'fontsize': 20})
        mask = (y == 0)
        s = 10
        axs[i, j].scatter(X[mask, 2 * i], X[mask, 2 * i + 1], s=s, alpha=0.5, color='salmon')
        axs[i, j].scatter(X[~mask, 2 * i], X[~mask, 2 * i + 1], s=s, alpha=0.5, color='royalblue')
        axs[i, j].set_xlabel(f'Feature {i * 2 + 1}', fontname='Century gothic')
        axs[i, j].set_ylabel(f'Feature {i * 2 + 2}', fontname='Century gothic')
    for i in range(n_features // 2, 3):
        axs[i, j].get_xaxis().set_visible(False)
        axs[i, j].get_yaxis().set_visible(False)
        for spine in axs[i, j].spines:
            axs[i, j].spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'datasets.eps'), transparent=True)


# Plot performance
results = {}
for dataset_name in ['xor', 'ring', 'ring+xor', 'ring+xor+sum']:
    results[dataset_name] = {}
    for method_name in METHOD_NAMES:
        results[dataset_name][method_name] = {}
        for n_features in [2, 4, 6, 8, 16, 32, 64, 128, 256, 512]:
            results[dataset_name][method_name][n_features] = {
                'auroc': np.nan,
                'auprc': np.nan,
                'best-k': np.nan,
                'best-2k': np.nan
            }

extra_method_names = ['cae', 'canceloutsigmoid', 'canceloutsoftmax', 'deeppinku', 'deeppink2o', 'fsnet', 'lassonet', 'mrmr', 'nnfs', 'relief', 'rf']
for method_name in extra_method_names:
    load_extra_results(os.path.join(RESULTS_PATH, f'{method_name}-xor.txt'), results['xor'])
load_nn_results(os.path.join(RESULTS_PATH, 'resultsXOR.txt'), results['xor'])
plot_performance(results['xor'], [4, 8, 16, 32, 64, 128, 256, 512], 2)
# plt.show()
plt.savefig(os.path.join(OUTPUT_PATH, 'benchmark-xor.png'), transparent=True)
plt.clf()

for method_name in extra_method_names:
    load_extra_results(os.path.join(RESULTS_PATH, f'{method_name}-ring.txt'), results['ring'])
load_nn_results(os.path.join(RESULTS_PATH, 'resultsRING.txt'), results['ring'])
plot_performance(results['ring'], [4, 8, 16, 32, 64, 128, 256, 512], 2)
# plt.show()
plt.savefig(os.path.join(OUTPUT_PATH, 'benchmark-ring.png'), transparent=True)
plt.clf()

for method_name in extra_method_names:
    load_extra_results(os.path.join(RESULTS_PATH, f'{method_name}-ring+xor.txt'), results['ring+xor'])
load_nn_results(os.path.join(RESULTS_PATH, 'resultsRING+XOR.txt'), results['ring+xor'])
plot_performance(results['ring+xor'], [4, 8, 16, 32, 64, 128, 256, 512], 4)
#plt.show()
plt.savefig(os.path.join(OUTPUT_PATH, 'benchmark-ring-xor.png'), transparent=True)
plt.clf()

for method_name in extra_method_names:
    load_extra_results(os.path.join(RESULTS_PATH, f'{method_name}-ring+xor+sum.txt'), results['ring+xor+sum'])
load_nn_results(os.path.join(RESULTS_PATH, 'resultsRING+XOR+SUM.txt'), results['ring+xor+sum'])
plot_performance(results['ring+xor+sum'], [6, 8, 16, 32, 64, 128, 256, 512], 6)
#plt.show()
plt.savefig(os.path.join(OUTPUT_PATH, 'benchmark-ring-xor-sum.png'), transparent=True)
plt.clf()

with open(os.path.join(TABLES_PATH, 'table.tex'), 'w') as f:
    f.write(create_table(results))
