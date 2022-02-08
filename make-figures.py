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
import matplotlib.pyplot as plt


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, 'data')
RESULTS_PATH = os.path.join(ROOT, 'results')
OUTPUT_PATH = os.path.join(ROOT, 'figures')
if not os.path.isdir(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

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
    'FeaturePermutation': ('Feature permutation', 'blue', 'D'),
    'ShapleyValueSampling': ('Shapley value sampling', 'mediumpurple', 'D'),
    'CancelOut_Sigmoid': ('CancelOut (sigmoid)', 'orangered', 'o'),
    'CancelOut_Softmax': ('CancelOut (softmax)', 'orange', 'o'),
    'DeepPINK_2o': ('DeepPINK (2o)', 'pink', 'x'),
    'DeepPINK_DK': ('DeepPINK (DK)', 'violet', 'x'),
    'CAE': ('Concrete Autoencoder', 'mediumseagreen', 'D'),
    'FSNet': ('FSNet', 'navy', 's'),
    'RF': ('Random Forest', 'darkgreen', 'D'),
    'Relief': ('Relief', 'turquoise', 'o')
}
METHOD_NAMES = list(METHODS.keys())


def load_pietro_results(filepath, results):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    header = lines[0].rstrip().split(',')
    lines = lines[1:]
    for line in lines:
        elements = line.rstrip().split(',')
        if len(elements) > 1:
            n_features = int(elements[1].replace('feat.csv', '').split('-')[-1])
            for i in range(2, len(elements)):
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


def plot_performance(results, ns):

    alpha = 0.6

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
    prop = {'family': 'Century gothic', 'size': 6}
    plt.legend(prop=prop, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, frameon=False)

    ax = plt.subplot(2, 2, 3)
    for method_name in METHOD_NAMES:
        label, color, marker = METHODS[method_name]
        ys = np.asarray([results[method_name][n_features]['best-k'] for n_features in ns])
        if not np.all(np.isnan(ys)):
            plt.plot(ns, 100 * ys, alpha=alpha, label=label, color=color, marker=marker)
    plt.xscale('log')
    plt.xlabel('Number of features', fontname='Century gothic')
    plt.ylabel('Best k (%)', fontname='Century gothic')
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
    plt.xscale('log')
    plt.xlabel('Number of features', fontname='Century gothic')
    plt.ylabel('Best 2k (%)', fontname='Century gothic')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n_features) for n_features in ns])
    prop = {'family': 'Century gothic', 'size': 6}
    plt.legend(prop=prop, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, frameon=False)

    #plt.tight_layout()


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

load_pietro_results(os.path.join(DATA_PATH, 'pietro', 'XOR_kfeat-2kfeat.csv'), results['xor'])
load_pietro_results(os.path.join(DATA_PATH, 'pietro', 'XOR_AUC-AUPRC.csv'), results['xor'])
load_nn_results(os.path.join(RESULTS_PATH, 'resultsXOR.txt'), results['xor'])
plot_performance(results['xor'], [2, 4, 8, 16, 32, 64, 128, 256, 512])
plt.savefig(os.path.join(OUTPUT_PATH, 'xor.png'), transparent=True)
plt.clf()

load_pietro_results(os.path.join(DATA_PATH, 'pietro', 'RING_kfeat-2kfeat.csv'), results['ring'])
load_pietro_results(os.path.join(DATA_PATH, 'pietro', 'RING_AUC-AUPRC.csv'), results['ring'])
load_nn_results(os.path.join(RESULTS_PATH, 'resultsRING.txt'), results['ring'])
plot_performance(results['ring'], [2, 4, 8, 16, 32, 64, 128, 256, 512])
plt.savefig(os.path.join(OUTPUT_PATH, 'ring.png'), transparent=True)
plt.clf()

load_pietro_results(os.path.join(DATA_PATH, 'pietro', 'RING+XOR_kfeat-2kfeat.csv'), results['ring+xor'])
load_pietro_results(os.path.join(DATA_PATH, 'pietro', 'RING+XOR_AUC-AUPRC.csv'), results['ring+xor'])
load_nn_results(os.path.join(RESULTS_PATH, 'resultsRING+XOR.txt'), results['ring+xor'])
plot_performance(results['ring+xor'], [2, 4, 8, 16, 32, 64, 128, 256, 512])
plt.savefig(os.path.join(OUTPUT_PATH, 'ring+xor.png'), transparent=True)
plt.clf()

load_pietro_results(os.path.join(DATA_PATH, 'pietro', 'RING+XOR+SUM_kfeat-2kfeat.csv'), results['ring+xor+sum'])
load_pietro_results(os.path.join(DATA_PATH, 'pietro', 'RING+XOR+SUM_AUC-AUPRC.csv'), results['ring+xor+sum'])
load_nn_results(os.path.join(RESULTS_PATH, 'resultsRING+XOR+SUM.txt'), results['ring+xor+sum'])
plot_performance(results['ring+xor+sum'], [2, 4, 8, 16, 32, 64, 128, 256, 512])
plt.savefig(os.path.join(OUTPUT_PATH, 'ring+xor+sum.png'), transparent=True)
plt.clf()
