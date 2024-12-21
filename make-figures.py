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

from src.data import generate_dataset

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
    'NN': ('Neural network', 'grey', 'x'),
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
    'DeepPINK': ('DeepPINK', 'pink', 'x'),
    'CAE': ('Concrete Autoencoder', 'mediumseagreen', 'D'),
    'FSNet': ('FSNet', 'navy', 's'),
    'RF': ('Random Forest', 'darkgreen', 'D'),
    'TreeSHAP': ('TreeSHAP', 'limegreen', 'D'),
    'Relief': ('Relief', 'turquoise', 'o'),
    'mi': ('Mutual information', 'cadetblue', '^'),
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
            n_features = int(elements[0].split('_')[-2])
            for i in range(1, len(elements)):
                if header[i].endswith('_AUC'):
                    results[header[i].replace('_AUC', '')][n_features]['auroc'] = float(elements[i])
                elif header[i].endswith('_TrainAUC'):
                    results[header[i].replace('_TrainAUC', '')][n_features]['train-auroc'] = float(elements[i])
                elif header[i].endswith('_AUPRC'):
                    results[header[i].replace('_AUPRC', '')][n_features]['auprc'] = float(elements[i])
                elif header[i].endswith('_TrainAUPRC'):
                    results[header[i].replace('_TrainAUPRC', '')][n_features]['train-auprc'] = float(elements[i])
                elif header[i].endswith('_bestK'):
                    results[header[i].replace('_bestK', '')][n_features]['best-k'] = float(elements[i])
                elif header[i].endswith('_best2K'):
                    results[header[i].replace('_best2K', '')][n_features]['best-2k'] = float(elements[i])
                elif header[i].endswith('_bestK2'):
                    results[header[i].replace('_bestK2', '')][n_features]['best-k2'] = float(elements[i])
                elif header[i].endswith('_best2K2'):
                    results[header[i].replace('_best2K2', '')][n_features]['best-2k2'] = float(elements[i])
                else:
                    raise NotImplementedError(header[i])


def plot_performance(results, ns, k):

    ns = np.asarray(ns)

    alpha = 0.5
    legend_size = 9

    plt.figure(figsize=(16, 8))

    ax = plt.subplot(2, 2, 1)
    for method_name in METHOD_NAMES:
        if method_name in 'TreeSHAP':
            continue
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
    perfs = []
    for method_name in METHOD_NAMES:
        if method_name in 'TreeSHAP':
            continue
        label, color, marker = METHODS[method_name]
        ys = np.asarray([results[method_name][n_features]['auprc'] for n_features in ns])
        if not np.all(np.isnan(ys)):
            plt.plot(ns, 100 * ys, alpha=alpha, label=label, color=color, marker=marker)
            perfs.append(np.mean(np.nan_to_num(ys)))
    plt.xscale('log')
    plt.ylabel('AUPRC (%)', fontname='Century gothic')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n_features) for n_features in ns])
    prop = {'family': 'Century gothic', 'size': legend_size}

    # Sort legend py performance
    handles, labels_ = plt.gca().get_legend_handles_labels()
    handles_order = np.argsort(perfs)[::-1]
    handles = [handles[idx] for idx in handles_order]
    labels_ = [labels_[idx] for idx in handles_order]
    plt.legend(handles, labels_, prop=prop, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, frameon=False)

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
    perfs = []
    for method_name in METHOD_NAMES:
        label, color, marker = METHODS[method_name]
        ys = np.asarray([results[method_name][n_features]['best-2k'] for n_features in ns])
        if not np.all(np.isnan(ys)):
            plt.plot(ns, 100 * ys, alpha=alpha, label=label, color=color, marker=marker)
            perfs.append(np.mean(np.nan_to_num(ys)))
    plt.fill_between(ns, np.minimum(100, 2 * 100 * k / ns), color='grey', alpha=0.25)
    plt.xscale('log')
    plt.xlabel('Number of features', fontname='Century gothic')
    plt.ylabel('Best 2p (%)', fontname='Century gothic')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n_features) for n_features in ns])

    # Sort legend py performance
    handles, labels_ = plt.gca().get_legend_handles_labels()
    handles_order = np.argsort(perfs)[::-1]
    handles = [handles[idx] for idx in handles_order]
    labels_ = [labels_[idx] for idx in handles_order]
    prop = {'family': 'Century gothic', 'size': legend_size}
    plt.legend(handles, labels_, prop=prop, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, frameon=False)

    plt.tight_layout()


def plot_performance_v2(results, ns, k):

    ns = np.asarray(ns)

    alpha = 0.5
    legend_size = 9

    plt.figure(figsize=(16, 8))

    for k, method_name in enumerate(METHOD_NAMES):
        ax = plt.subplot(6, 4, k + 1)
        label, color, marker = METHODS[method_name]
        ys = np.asarray([results[method_name][n_features]['best-k'] for n_features in ns])
        if not np.all(np.isnan(ys)):
            ax.plot(ns, 100 * ys, alpha=alpha, label=label, color=color, marker=marker, markersize=4)
        ax.fill_between(ns, np.minimum(100, 100 * k / ns), color='grey', alpha=0.25)
        ax.set_xscale('log')
        ax.set_xlabel('Number of features', fontname='Century gothic')
        ax.set_ylabel('Best 2p (%)', fontname='Century gothic')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n_features) for n_features in ns])
    #prop = {'family': 'Century gothic', 'size': legend_size}
    #plt.legend(prop=prop, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, frameon=False)
    # plt.tight_layout()


def plot_fa_bars(results, results_b, results_t):
    color1, color2, color3 = 'darkcyan', 'royalblue', 'purple'
    plt.figure(figsize=(8, 8))
    for i, method_name in enumerate(fa_method_names):
        ax = plt.subplot(4, 3, i + 1)
        alpha = 0.6
        label, color, marker = METHODS[method_name]
        metric = 'best-2k'

        ax.set_title(label, fontsize=6)

        ns = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        dataset_name = 'xor'
        ys = np.asarray([results[dataset_name][method_name][n_features][metric] for n_features in ns])
        plt.bar(-0.5, 100 * np.mean(ys), color=color1, width=0.4)
        ys = np.asarray([results_b[dataset_name][method_name][n_features][metric] for n_features in ns])
        plt.bar(0, 100 * np.mean(ys), color=color2, width=0.4)
        ys = np.asarray([results_t[dataset_name][method_name][n_features][metric] for n_features in ns])
        plt.bar(0.5, 100 * np.mean(ys), color=color3, width=0.4)

        ns = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        dataset_name = 'xor'
        ys = np.asarray([results[dataset_name][method_name][n_features][metric] for n_features in ns])
        plt.bar(2.5, 100 * np.mean(ys), color=color1, width=0.4)
        ys = np.asarray([results_b[dataset_name][method_name][n_features][metric] for n_features in ns])
        plt.bar(3, 100 * np.mean(ys), color=color2, width=0.4)
        ys = np.asarray([results_t[dataset_name][method_name][n_features][metric] for n_features in ns])
        plt.bar(3.5, 100 * np.mean(ys), color=color3, width=0.4)

        ns = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        dataset_name = 'ring+xor'
        ys = np.asarray([results[dataset_name][method_name][n_features][metric] for n_features in ns])
        plt.bar(5.5, 100 * np.mean(ys), color=color1, width=0.4)
        ys = np.asarray([results_b[dataset_name][method_name][n_features][metric] for n_features in ns])
        plt.bar(6, 100 * np.mean(ys), color=color2, width=0.4)
        ys = np.asarray([results_t[dataset_name][method_name][n_features][metric] for n_features in ns])
        plt.bar(6.5, 100 * np.mean(ys), color=color3, width=0.4)

        ns = [6, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        dataset_name = 'ring+xor+sum'
        ys = np.asarray([results[dataset_name][method_name][n_features][metric] for n_features in ns])
        plt.bar(8.5, 100 * np.mean(ys), color=color1, width=0.4)
        ys = np.asarray([results_b[dataset_name][method_name][n_features][metric] for n_features in ns])
        plt.bar(9, 100 * np.mean(ys), color=color2, width=0.4)
        ys = np.asarray([results_t[dataset_name][method_name][n_features][metric] for n_features in ns])
        plt.bar(9.5, 100 * np.mean(ys), color=color3, width=0.4)

        ns = [2000]
        dataset_name = 'dag'
        ys = np.asarray([results[dataset_name][method_name][n_features][metric] for n_features in ns])
        plt.bar(11.5, 100 * np.mean(ys), color=color1, width=0.4)
        ys = np.asarray([results_b[dataset_name][method_name][n_features][metric] for n_features in ns])
        plt.bar(12, 100 * np.mean(ys), color=color2, width=0.4)
        ys = np.asarray([results_t[dataset_name][method_name][n_features][metric] for n_features in ns])
        plt.bar(12.5, 100 * np.mean(ys), color=color3, width=0.4)

        plt.ylabel('Best 2p (%)', fontname='Century gothic', fontsize=6)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xticks([0, 3, 6, 9, 12])
        ax.tick_params(axis='y', labelsize=6)
        ax.set_xticklabels(['XOR', 'RING', 'RING+XOR', 'R+X+S', 'DAG'], fontsize=6)

    ax = plt.subplot(4, 3, 11)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.bar(11.5, 0, color=color1, width=0.4, label='Validation set')
    ax.bar(12, 0, color=color2, width=0.4, label='Validation set + Bootstrapping')
    ax.bar(12.5, 0, color=color3, width=0.4, label='Training set')
    plt.legend(bbox_to_anchor=(1.2, 1), borderaxespad=0, prop={'size': 6})
    plt.tight_layout()


def create_table(results):
    s = '\\begin{table}[h!]\n'
    s += '\\centering\n'
    s += '\\label{tab:benchmark}{\\resizebox{\\columnwidth}{!}{\\begin{tabular}{lrrrrrrrrrr}\\toprule\n'
    s += 'Dataset & \\multicolumn{2}{c}{\\ring} & \\multicolumn{2}{c}{\\xor} & \\multicolumn{2}{c}{\\ringxor} & \\multicolumn{2}{c}{\\ringxorsum} & \\multicolumn{2}{c}{DAG} \\\\\n'
    s += '\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-9} \\cmidrule(lr){10-11}\n'
    s += 'Method & Best p & Best 2p & Best p & Best 2p & Best p & Best 2p & Best p & Best 2p & Best p & Best 2p \\\\ \\midrule\n'

    def f(method_name):
        s = METHODS[method_name][0]
        for dataset_name in ['ring', 'xor', 'ring+xor', 'ring+xor+sum', 'dag']:
            data = [results[dataset_name][method_name][n_features]['best-k'] for n_features in results[dataset_name][method_name].keys()]
            best_k = 0 if (len(data) == 0) else 100 * np.nanmean(data)
            data = [results[dataset_name][method_name][n_features]['best-2k'] for n_features in results[dataset_name][method_name].keys()]
            best_2k = 0 if (len(data) == 0) else 100 * np.nanmean(data)
            s += f' & {best_k:.1f} & {best_2k:.1f}'
        s += ' \\\\\n'
        return s

    for method_name in ['Saliency', 'IG_noMul', 'DeepLift', 'InputXGradient', 'SmoothGrad', 'GuidedBackprop',
                        'Deconvolution', 'FeatureAblation', 'FeaturePermutation', 'ShapleyValueSampling']:
        s += f(method_name)
    s += '\\midrule\n'
    for method_name in ['mi', 'mRMR', 'LassoNet', 'Relief', 'CAE', 'FSNet', 'CancelOut_Softmax', 'CancelOut_Sigmoid',
                        'DeepPINK', 'RF', 'TreeSHAP']:
        s += f(method_name)

    s += '\\bottomrule\n'
    s += '\\end{tabular}}}{}\n'
    s += '\\caption{Best p and best 2p score percentages on the 5 datasets. For the first 4 datasets, scores have been averaged over $m \\in \\{2, 4, 6, 8, 16, 32, 64, 128, 256, 512, 1024, 2048\\}$.'
    s += 'Top and bottom parts of the table correspond to instance-level feature attribution and embedded/filter FS methods, respectively. Best performing methods are highlighted in bold.}\n'
    s += '\\end{table}\n'
    return s


def create_dag_table(results):
    s = '\\begin{table}[h!]\n'
    s += '\\centering\n'
    s += '\\label{tab:benchmark-dag}{\\resizebox{\\columnwidth}{!}{\\begin{tabular}{lrrrrrr}\\toprule\n'
    s += 'Dataset & \\multicolumn{2}{c}{\% causal features} & \\multicolumn{2}{c}{\% correlated features} & \\multicolumn{2}{c}{Performance (\%)} \\\\\n'
    s += '\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}\n'
    s += 'Method & Best p & Best 2p & Best p & Best 2p & AUROC & AUPRC \\\\ \\midrule\n'

    def f(method_name):
        s = METHODS[method_name][0]
        dataset_name = 'dag'
        n_features = 2000
        best_k = 100 * results[dataset_name][method_name][n_features]['best-k']
        best_2k = 100 * results[dataset_name][method_name][n_features]['best-2k']
        best_k2 = 100 * results[dataset_name][method_name][n_features]['best-k2']
        best_2k2 = 100 * results[dataset_name][method_name][n_features]['best-2k2']
        auroc = 100 * results[dataset_name][method_name][n_features]['auroc']
        auprc = 100 * results[dataset_name][method_name][n_features]['auprc']
        if np.isnan(auroc) or (method_name.lower() == 'treeshap'):
            s += f' & {best_k:.1f} & {best_2k:.1f} & {best_k2:.1f} & {best_2k2:.1f} & - & -'
        else:
            s += f' & {best_k:.1f} & {best_2k:.1f} & {best_k2:.1f} & {best_2k2:.1f} & {auroc:.1f} & {auprc:.1f}'

        s += ' \\\\\n'
        return s

    for method_name in ['Saliency', 'IG_noMul', 'DeepLift', 'InputXGradient', 'SmoothGrad', 'GuidedBackprop',
                        'Deconvolution', 'FeatureAblation', 'FeaturePermutation', 'ShapleyValueSampling']:
        s += f(method_name)
    s += '\\midrule\n'
    for method_name in ['mi', 'mRMR', 'LassoNet', 'Relief', 'CAE', 'FSNet', 'CancelOut_Softmax', 'CancelOut_Sigmoid',
                        'DeepPINK', 'RF', 'TreeSHAP']:
        s += f(method_name)

    s += '\\bottomrule\n'
    s += '\\end{tabular}}}{}\n'
    s += '\\caption{Best p and best 2p scores of all feature selection methods on the DAG dataset. $p=7$ when considering only causal features ' \
         'as relevant (first multi-column) and $p=81$ when also including confouders (second multi-column). ' \
         'Additionally, AUROC and AUPRC scores have been reported for embedded methods.}\n'
    s += '\\end{table}\n'
    return s


# Plot performance
results = {}
results_b = {}
results_t = {}
for dataset_name in ['xor', 'ring', 'ring+xor', 'ring+xor+sum', 'dag']:
    if dataset_name == 'dag':
        ns = [2000]
    else:
        ns = [2, 4, 6, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    results[dataset_name] = {}
    results_b[dataset_name] = {}
    results_t[dataset_name] = {}
    for method_name in METHOD_NAMES:
        results[dataset_name][method_name] = {}
        results_b[dataset_name][method_name] = {}
        results_t[dataset_name][method_name] = {}
        for n_features in ns:
            results[dataset_name][method_name][n_features] = {'auroc': np.nan, 'auprc': np.nan, 'best-k': np.nan, 'best-2k': np.nan}
            results_b[dataset_name][method_name][n_features] = {'auroc': np.nan, 'auprc': np.nan, 'best-k': np.nan, 'best-2k': np.nan}
            results_t[dataset_name][method_name][n_features] = {'auroc': np.nan, 'auprc': np.nan, 'best-k': np.nan, 'best-2k': np.nan}

fa_method_names = [
    'Saliency', 'InputXGradient', 'IG_noMul',
    'GuidedBackprop', 'FeaturePermutation', 'FeatureAblation',
    'Deconvolution', 'DeepLift', 'SmoothGrad',
    'ShapleyValueSampling'
]
n_samples = 1000
for dataset_name in ['dag', 'xor', 'ring', 'ring+xor', 'ring+xor+sum']:
    for method_name in fa_method_names:
        print(f'Loading {method_name}-{dataset_name}-{n_samples}.txt')
        load_extra_results(os.path.join(RESULTS_PATH, f'{method_name}-{dataset_name}-{n_samples}.txt'), results[dataset_name])
        print(f'Loading {method_name}-b-{dataset_name}-{n_samples}.txt')
        load_extra_results(os.path.join(RESULTS_PATH, f'{method_name}-b-{dataset_name}-{n_samples}.txt'), results_b[dataset_name])
        print(f'Loading {method_name}-t-{dataset_name}-{n_samples}.txt')
        load_extra_results(os.path.join(RESULTS_PATH, f'{method_name}-t-{dataset_name}-{n_samples}.txt'), results_t[dataset_name])
plot_fa_bars(results, results_b, results_t)
plt.savefig(os.path.join(OUTPUT_PATH, 'bootstrapping.png'), transparent=True, dpi=400)
plt.clf()



extra_method_names = [
    'nn', 'treeshap', 'canceloutsigmoid', 'canceloutsoftmax', 'deeppink', 'fsnet', 'mi', 'mrmr', 'relief', 'rf', 'lassonet', 'cae',
    'InputXGradient', 'IG_noMul', 'GuidedBackprop', 'FeaturePermutation', 'FeatureAblation', 'Deconvolution', 'DeepLift',
    'SmoothGrad', 'ShapleyValueSampling', 'Saliency'
]

"""
method_name = 'nn'
dataset_name = 'xor'
train_aurocs, aurocs = [], []
ns = np.asarray([50, 100, 250, 500, 1000, 2000, 5000, 10000, 20000, 50000])
for n_samples in ns:
    load_extra_results(os.path.join(RESULTS_PATH, f'{method_name}-{dataset_name}-{n_samples}.txt'), results[dataset_name])
    train_aurocs.append(results[dataset_name]['NN'][2048]['train-auroc'])
    aurocs.append(results[dataset_name]['NN'][2048]['auroc'])
plt.figure(figsize=(16, 8))
ax = plt.subplot(1, 1, 1)
plt.plot(ns, train_aurocs, label='Train')
plt.plot(ns, aurocs, label='Test')
plt.ylabel('AUC-ROC', fontname='Century gothic')
plt.xlabel('Dataset size (XOR)', fontname='Century gothic')
ax.set_xscale('log')
ax.set_xticks(ns)
ax.set_xticklabels([str(n_features) for n_features in ns])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend()
plt.show()
"""

# extra_method_names = ['treeshap', 'cae', 'canceloutsigmoid', 'canceloutsoftmax', 'deeppink', 'fsnet', 'lassonet', 'mrmr', 'relief', 'rf']
for n_samples in [250, 500, 1000]:
    for method_name in extra_method_names:
        print(f'Loading {method_name}-dag-{n_samples}.txt')
        load_extra_results(os.path.join(RESULTS_PATH, f'{method_name}-dag-{n_samples}.txt'), results['dag'])

    for method_name in extra_method_names:
        print(f'Loading {method_name}-xor-{n_samples}.txt')
        load_extra_results(os.path.join(RESULTS_PATH, f'{method_name}-xor-{n_samples}.txt'), results['xor'])
    plot_performance(results['xor'], [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048], 2)
    # plt.show()
    plt.savefig(os.path.join(OUTPUT_PATH, f'benchmark-xor-{n_samples}.png'), transparent=True)
    plt.clf()

    for method_name in extra_method_names:
        print(f'Loading {method_name}-ring-{n_samples}.txt')
        load_extra_results(os.path.join(RESULTS_PATH, f'{method_name}-ring-{n_samples}.txt'), results['ring'])
    plot_performance(results['ring'], [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048], 2)
    # plt.show()
    plt.savefig(os.path.join(OUTPUT_PATH, f'benchmark-ring-{n_samples}.png'), transparent=True)
    plt.clf()

    for method_name in extra_method_names:
        print(f'Loading {method_name}-ring+xor-{n_samples}.txt')
        load_extra_results(os.path.join(RESULTS_PATH, f'{method_name}-ring+xor-{n_samples}.txt'), results['ring+xor'])
    plot_performance(results['ring+xor'], [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048], 4)
    #plt.show()
    plt.savefig(os.path.join(OUTPUT_PATH, f'benchmark-ring-xor-{n_samples}.png'), transparent=True)
    plt.clf()

    for method_name in extra_method_names:
        print(f'Loading {method_name}-ring+xor+sum-{n_samples}.txt')
        load_extra_results(os.path.join(RESULTS_PATH, f'{method_name}-ring+xor+sum-{n_samples}.txt'), results['ring+xor+sum'])
    plot_performance(results['ring+xor+sum'], [6, 8, 16, 32, 64, 128, 256, 512, 1024, 2048], 6)
    #plt.show()
    plt.savefig(os.path.join(OUTPUT_PATH, f'benchmark-ring+xor+sum-{n_samples}.png'), transparent=True)
    plt.clf()

    with open(os.path.join(TABLES_PATH, f'table-{n_samples}.tex'), 'w') as f:
        f.write(create_table(results))

    with open(os.path.join(TABLES_PATH, f'table-dag-{n_samples}.tex'), 'w') as f:
        f.write(create_dag_table(results))
