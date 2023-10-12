# -*- coding: utf-8 -*-
#
#  main.py
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

import argparse
import os

import captum.attr
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from src.core import run_fs_method
from src.dag import load_dag_dataset
from src.data import generate_dataset
from src.nn_wrapper import NNwrapper


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, 'data')
RESULTS_PATH = os.path.join(ROOT, 'results')


FS_METHODS = {
    'nn': ('NN', False, True),
    'nnfs': ('Saliency', False, True),
    'canceloutsigmoid': ('CancelOut_Sigmoid', True, True),
    'canceloutsoftmax': ('CancelOut_Softmax', True, True),
    'deeppink': ('DeepPINK', True, True),
    'rf': ('RF', True, True),
    'treeshap': ('TreeSHAP', True, True),
    'relief': ('Relief', True, False),
    'fsnet': ('FSNet', True, True),
    'mrmr': ('mRMR', True, False),
    'mi': ('MI', True, False),
    'lassonet': ('LassoNet', True, True),
    'cae': ('CAE', True, True)
}
SA_METHODS = {
    'Saliency': 'Saliency',
    'InputXGradient': 'Input x Gradient',
    'IG_noMul': 'Integrated gradient',
    'SmoothGrad': 'SmoothGrad',
    'GuidedBackprop': 'Guided backpropagation',
    'DeepLift': 'DeepLift',
    'Deconvolution': 'Deconvolution',
    'FeatureAblation': 'Feature ablation',
    'FeaturePermutation': 'Feature permutation',
    'ShapleyValueSampling': 'Shapley value sampling'
}
SA_METHOD_NAMES = list(SA_METHODS.keys())


def run_fa_methods(wrapper, X_test, method_name):
    tmp_x = torch.FloatTensor(X_test)
    tmp_x.requires_grad_()
    baselines = torch.zeros((1, tmp_x.size()[-1]))
    if 'IG_noMul' == method_name:
        ig = captum.attr.IntegratedGradients(wrapper.model, multiply_by_inputs=False)
        attr = ig.attribute(tmp_x, target=0, return_convergence_delta=False, baselines=baselines)
    elif 'Saliency' == method_name:
        ig = captum.attr.Saliency(wrapper.model)
        attr = ig.attribute(tmp_x, target=0, abs=True)
    elif 'DeepLift' == method_name:
        ig = captum.attr.DeepLift(wrapper.model, multiply_by_inputs=False)
        attr = ig.attribute(tmp_x, target=0, return_convergence_delta=False, baselines=baselines)
    elif 'InputXGradient' == method_name:
        ig = captum.attr.InputXGradient(wrapper.model)
        attr = ig.attribute(tmp_x, target=0)
    elif 'SmoothGrad' == method_name:
        ig = captum.attr.NoiseTunnel(captum.attr.Saliency(wrapper.model))
        attr = ig.attribute(tmp_x, target=0, nt_samples=50, stdevs=0.1)
    elif 'GuidedBackprop' == method_name:
        ig = captum.attr.GuidedBackprop(wrapper.model)
        attr = ig.attribute(tmp_x, target=0)
    elif 'Deconvolution' == method_name:
        ig = captum.attr.Deconvolution(wrapper.model)
        attr = ig.attribute(tmp_x, target=0)
    elif 'FeatureAblation' == method_name:
        ig = captum.attr.FeatureAblation(wrapper.model)
        attr = ig.attribute(tmp_x, target=0, baselines=baselines)
    elif 'FeaturePermutation' == method_name:
        ig = captum.attr.FeaturePermutation(wrapper.model)
        attr = ig.attribute(tmp_x, target=0)
    elif 'ShapleyValueSampling' == method_name:
        ig = captum.attr.ShapleyValueSampling(wrapper.model)
        attr = ig.attribute(tmp_x, target=0, baselines=baselines)
    else:
        raise NotImplementedError(method_name)

    attr = attr.detach().numpy()
    attr = np.mean(np.abs(attr), axis=0)
    return attr


def run_method(splits, X, X_tilde, y, method_name, dataset_name, k, k2=None, h_scores=None):


    n_features = X.shape[1]

    best_k = []
    best_2k = []
    best_k2 = []
    best_2k2 = []
    train_aurocs = []
    train_auprcs = []
    aurocs = []
    auprcs = []
    for i in range(len(splits)):
        train_index, test_index = splits[i]
        X_train, X_test = X[train_index], X[test_index]
        X_tilde_train, X_tilde_test = X_tilde[train_index], X_tilde[test_index]
        y_train, y_test = y[train_index], y[test_index]
        assert len(X_train) >= len(X_test)

        # Randomly permuting features
        idx = np.arange(X.shape[1])
        np.random.shuffle(idx)
        X_train, X_test = X_train[:, idx], X_test[:, idx]
        X_tilde_train, X_tilde_test = X_tilde_train[:, idx], X_tilde_test[:, idx]
        correct_indices = set(list(np.where(idx < k)[0]))
        if k2 is None:
            correct_indices2 = None
        else:
            correct_indices2 = set(list(np.where(idx < k2)[0]))

        y_hat = None
        y_train_hat = None

        if h_scores is not None:
            scores = h_scores[i][idx]
            scores2 = scores
        else:
            y_train_hat, y_hat, scores, scores2 = run_fs_method(
                dataset_name, method_name, X_train, X_tilde_train,
                y_train, X_test, X_tilde_test, k
            )
            y_train = y_train
            y_train_hat = y_train_hat

        if scores is not None:
            assert scores.shape == (n_features,)
            indices = np.argsort(np.abs(scores))
            best_k.append(np.sum([i in correct_indices for i in indices[-k:]]) / k)
            indices = np.argsort(np.abs(scores2))
            best_2k.append(np.sum([i in correct_indices for i in indices[-2 * k:]]) / k)
            if k2 is not None:
                indices = np.argsort(np.abs(scores))
                best_k2.append(np.sum([i in correct_indices2 for i in indices[-k2:]]) / k2)
                indices = np.argsort(np.abs(scores2))
                best_2k2.append(np.sum([i in correct_indices2 for i in indices[-2 * k2:]]) / k2)
        if y_hat is not None:
            aurocs.append(roc_auc_score(y_test, y_hat))
            auprcs.append(average_precision_score(y_test, y_hat))
        if y_train_hat is not None:
            train_aurocs.append(roc_auc_score(y_train, y_train_hat))
            train_auprcs.append(average_precision_score(y_train, y_train_hat))
    return best_k, best_2k, best_k2, best_2k2, train_aurocs, train_auprcs, aurocs, auprcs


def process_dataset_(method_name, dataset_name, k, ns, n_samples=1000):

    with open(os.path.join(RESULTS_PATH, f'{method_name}-{dataset_name}-{n_samples}.txt'), 'w') as f:

        method, does_fs, has_classifier = FS_METHODS[method_name]

        row = f'Dataset'
        if does_fs:
            row += f'\t{method}_bestK\t{method}_best2K'
            if dataset_name == 'dag':
                row += f'\t{method}_bestK2\t{method}_best2K2'
        if has_classifier:
            row += f'\t{method}_TrainAUC\t{method}_TrainAUPRC\t{method}_AUC\t{method}_AUPRC'
        row += f'\n'
        f.write(row)
        for n_features in ns:

            if dataset_name == 'dag':
                X, X_tilde, y, k, k2 = load_dag_dataset(os.path.join(ROOT, 'data'))
                if n_samples != len(X):
                    idx = np.arange(len(X))
                    np.random.shuffle(idx)
                    X, X_tilde, y = X[idx], X_tilde[idx], y[idx]
                X = StandardScaler().fit_transform(X)
                X_tilde = StandardScaler().fit_transform(X_tilde)
            else:
                X, X_tilde, y = generate_dataset(dataset_name, n_samples, n_features)
                k2 = None
                X = 2. * X - 1.
                X_tilde = 2. * X_tilde - 1.

            splits = list(KFold(n_splits=6).split(X))
            best_k, best_2k, best_k2, best_2k2, train_aurocs, train_auprcs, aurocs, auprcs = run_method(
                splits, X, X_tilde, y, method_name, dataset_name, k, k2=k2)

            if len(best_k) > 0:
                print(f'Average best-k: {np.mean(best_k)}')
                print(f'Average best-2k: {np.mean(best_2k)}')
            if len(best_k2) > 0:
                print(f'Average best-k2: {np.mean(best_k2)}')
                print(f'Average best-2k2: {np.mean(best_2k2)}')
            if len(train_aurocs) > 0:
                print(f'Average Training AUROC: {np.mean(train_aurocs)}')
                print(f'Average Training AUPRC: {np.mean(train_auprcs)}')
            if len(aurocs) > 0:
                print(f'Average AUROC: {np.mean(aurocs)}')
                print(f'Average AUPRC: {np.mean(auprcs)}')
            row_name = f'{dataset_name}_{n_features}_{n_samples}'
            row = f'{row_name}'
            if does_fs:
                row += f'\t{np.mean(best_k)}\t{np.mean(best_2k)}'
                if dataset_name == 'dag':
                    row += f'\t{np.mean(best_k2)}\t{np.mean(best_2k2)}'
            if has_classifier:
                row += f'\t{np.mean(train_aurocs)}\t{np.mean(train_auprcs)}\t{np.mean(aurocs)}\t{np.mean(auprcs)}'
            row += '\n'
            f.write(row)


def process_dataset_with_fa_methods(dataset_name, k, ns, n_samples=1000, bootstrap=False, with_train=False):
    splits = list(KFold(n_splits=6).split(np.random.rand(n_samples)))

    models = {(n_features, j): NNwrapper.create(dataset_name, n_features) for n_features in ns for j in range(6)}
    for j in range(6):
        for n_features in ns:
            wrappers = []
            if bootstrap:
                for _ in range(10):
                    wrappers.append(NNwrapper.create(dataset_name, n_features))
            else:
                wrappers.append(NNwrapper.create(dataset_name, n_features))
            models[(n_features, j)] = wrappers

    for method_name in SA_METHOD_NAMES:
        if bootstrap and (not with_train):
            filename = f'{method_name}-b-{dataset_name}-{n_samples}.txt'
        elif with_train and (not bootstrap):
            filename = f'{method_name}-t-{dataset_name}-{n_samples}.txt'
        elif with_train and bootstrap:
            raise NotImplementedError
        else:
            filename = f'{method_name}-{dataset_name}-{n_samples}.txt'
        with open(os.path.join(RESULTS_PATH, filename), 'w') as f:

            row = f'Dataset'
            row += f'\t{method_name}_bestK\t{method_name}_best2K'
            if dataset_name == 'dag':
                row += f'\t{method_name}_bestK2\t{method_name}_best2K2'
            row += f'\n'
            f.write(row)
            for n_features in ns:

                if dataset_name == 'dag':
                    X, X_tilde, y, k, k2 = load_dag_dataset(os.path.join(ROOT, 'data'))
                else:
                    X, X_tilde, y = generate_dataset(dataset_name, n_samples, n_features)
                    k2 = None

                # Centering the data
                X = 2. * X - 1.
                X_tilde = 2. * X_tilde - 1.

                h_scores = []
                for j in range(len(splits)):
                    train_index, test_index = splits[j]
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    hr_scores = []
                    for wrapper in models[(n_features, j)]:
                        if not wrapper.trained:
                            if bootstrap:
                                idx = np.random.randint(0, len(X_train), size=int(0.8 * len(X_train)))
                                wrapper.fit(X_train[idx, :], y_train[idx])
                            else:
                                wrapper.fit(X_train, y_train)
                        if with_train:
                            hr_scores.append(run_fa_methods(wrapper, X_train, method_name))
                        else:
                            hr_scores.append(run_fa_methods(wrapper, X_test, method_name))
                    hr_scores = np.asarray(hr_scores)
                    h_scores.append(np.mean(hr_scores, axis=0))

                best_k, best_2k, best_k2, best_2k2, train_aurocs, train_auprcs, aurocs, auprcs = run_method(
                    splits, X, X_tilde, y, method_name, dataset_name, k, k2=k2, h_scores=h_scores)

                if len(best_k) > 0:
                    print(f'Average best-k: {np.mean(best_k)}')
                    print(f'Average best-2k: {np.mean(best_2k)}')
                if len(best_k2) > 0:
                    print(f'Average best-k2: {np.mean(best_k2)}')
                    print(f'Average best-2k2: {np.mean(best_2k2)}')
                row_name = f'{dataset_name}_{n_features}_{n_samples}'
                row = f'{row_name}'
                row += f'\t{np.mean(best_k)}\t{np.mean(best_2k)}'
                if dataset_name == 'dag':
                    row += f'\t{np.mean(best_k2)}\t{np.mean(best_2k2)}'
                row += '\n'
                f.write(row)


def process_dataset(method_name, dataset_name, k, ns, n_samples=1000):
    if method_name == 'attr':
        process_dataset_with_fa_methods(
            dataset_name, k, ns, n_samples=n_samples, bootstrap=False, with_train=False)
    elif method_name == 'attr-t':
        process_dataset_with_fa_methods(
            dataset_name, k, ns, n_samples=n_samples, bootstrap=False, with_train=True)
    elif method_name == 'attr-b':
        process_dataset_with_fa_methods(
            dataset_name, k, ns, n_samples=n_samples, bootstrap=True, with_train=False)
    else:
        process_dataset_(method_name, dataset_name, k, ns, n_samples=n_samples)


if __name__ == '__main__':

    method_names = [
        'attr', 'attr-t', 'attr-b',
        'nn', 'rf', 'relief', 'fsnet', 'mrmr', 'mi', 'lassonet', 'cae',
        'treeshap', 'canceloutsigmoid', 'canceloutsoftmax', 'deeppink']
    parser = argparse.ArgumentParser()
    parser.add_argument('method', type=str, choices=method_names, help='Method name')
    args = parser.parse_args()

    for n_samples in [1000]:
        ns = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        process_dataset(args.method, 'ring+xor+sum', 6, [6] + ns, n_samples=n_samples)
        process_dataset(args.method, 'ring+xor', 4, [4] + ns, n_samples=n_samples)
        process_dataset(args.method, 'ring', 2, [2, 4] + ns, n_samples=n_samples)
        process_dataset(args.method, 'xor', 2, [2, 4] + ns, n_samples=n_samples)
        process_dataset(args.method, 'dag', 6, [2000], n_samples=n_samples)
