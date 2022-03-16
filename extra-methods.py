# -*- coding: utf-8 -*-
#
#  extra-methods.py
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

import numpy as np
import pandas as pd
import torch

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from src import nn_wrapper
from src.fsnet import FSNet
from src.relief import relief


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, 'data')
RESULTS_PATH = os.path.join(ROOT, 'results')

SEED = 0xCAFE


FS_METHODS = {
    'nn': ('NN', False, True),
    'nnfs': ('Saliency', False, True),
    'canceloutsigmoid': ('CancelOut_Sigmoid', True, True),
    'canceloutsoftmax': ('CancelOut_Softmax', True, True),
    'deeppink2o': ('DeepPINK_2o', True, True),
    'deeppinku': ('DeepPINK_U', True, True),
    'rf': ('RF', True, True),
    'relief': ('Relief', True, False),
    'fsnet': ('FSNet', True, True),
    'mrmr': ('mRMR', True, False),
    'lassonet': ('LassoNet', True, True),
    'cae': ('CAE', True, True)
}


def load_indices(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    idx = []
    for line in lines:
        line = line.rstrip()
        if len(line) > 1:
            idx.append(int(float(line)))
    return np.asarray(idx)


def process_dataset(method_name, dataset_name, k, ns):

    knockoff_features = np.load(os.path.join(DATA_PATH, 'knock-off-features-2o.npy'))

    with open(os.path.join(RESULTS_PATH, f'{method_name}-{dataset_name}.txt'), 'w') as f:

        method, does_fs, has_classifier = FS_METHODS[method_name]
        if does_fs and has_classifier:
            f.write(f'Dataset\t{method}_bestK\t{method}_best2K\t{method}_AUC\t{method}_AUPRC\n')
        elif does_fs:
            f.write(f'Dataset\t{method}_bestK\t{method}_best2K\n')
        elif has_classifier:
            f.write(f'Dataset\t{method}_AUC\t{method}_AUPRC\n')
        else:
            raise NotImplementedError()
        for n_features in ns:
            if dataset_name != 'ring+xor+sum':
                filename = f'{dataset_name}_1000samples-{n_features}feat.csv'
            else:
                filename = f'ring-xor-sum_1000samples-{n_features}feat.csv'
            filepath = os.path.join(DATA_PATH, filename)
            column_names = [f'x{i}' for i in range(n_features)] + ['y']
            df = pd.read_csv(filepath, delimiter=',', header=None, names=column_names)
            y = df['y'].to_numpy().astype(int)
            df.drop('y', axis=1, inplace=True)
            X = df.to_numpy().astype(np.float32)

            # Centering the data
            X = 2. * X - 1.
            assert np.all(np.logical_and(X >= -1, X <= 1))

            best_k = []
            best_2k = []
            aurocs = []
            auprcs = []
            for i in range(6):
                print(dataset_name, n_features, i)
                idx = load_indices(os.path.join(DATA_PATH, f'Indices_fold{i}.csv')).astype(int)
                mask = np.zeros(len(X), dtype=bool)
                mask[idx] = True
                X_train, y_train = X[~mask], y[~mask]
                X_test, y_test = X[mask], y[mask]
                knockoff_train = X[~mask, :n_features]
                knockoff_test = X[mask, :n_features]
                assert len(X_train) >= len(X_test)

                # Randomly permuting features
                idx = np.arange(X.shape[1])
                np.random.shuffle(idx)
                X_train, X_test = X_train[:, idx], X_test[:, idx]
                correct_indices = set(list(np.where(idx < k)[0]))

                y_hat = None
                scores = None
                scores2 = None

                if method_name == 'nn':
                    wrapper = nn_wrapper.NNwrapper.create(dataset_name, len(X[0]))
                    wrapper.fit(X_train, y_train)
                    y_hat = wrapper.predict(X_test)
                elif method_name == 'nnfs':
                    wrapper = nn_wrapper.NNwrapper.create(dataset_name, len(X[0]))
                    wrapper.fit(X_train, y_train)
                    scores = np.abs(wrapper.feature_importance(X_train))
                    scores = scores2
                    indices = np.argsort(scores)
                    indices = indices[-2*k:]
                    wrapper = nn_wrapper.NNwrapper.create(dataset_name, len(indices))
                    wrapper.fit(X_train[:, indices], y_train)
                    y_hat = wrapper.predict(X_test[:, indices])
                elif method_name == 'rf':
                    clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=SEED)
                    clf.fit(X_train, y_train)
                    scores = clf.feature_importances_
                    scores2 = scores
                    y_hat = clf.predict_proba(X_test)[:, 1]
                elif method_name == 'relief':
                    scores = relief(X_train, y_train)
                    scores2 = scores
                elif method_name == 'fsnet':
                    n_selected = min(2 * k, X_train.shape[1])
                    fsnet = FSNet(nn_wrapper.Model(n_selected), X.shape[1], 30, n_selected)
                    fsnet.fit(X_train, y_train)
                    y_hat = fsnet.predict(X_test)
                    scores = fsnet.get_feature_importances()
                    scores2 = scores
                elif method_name == 'lassonet':
                    import lassonet
                    model = lassonet.LassoNetClassifier(
                        hidden_dims=(64, 64),
                        n_iters=(200, 100),
                        batch_size=64,
                        dropout=0.2,
                        patience=10,
                        tol=0.9999,
                        verbose=False)
                    path = model.path(X_train, y_train)
                    scores = model.feature_importances_.numpy()
                    scores2 = scores
                    val_losses = [save.val_loss for save in path]
                    state_dict = path[np.argmin(val_losses)].state_dict
                    model.load(state_dict)
                    y_hat = model.predict(X_test)
                elif method_name == 'mrmr':
                    import pymrmr
                    n_bins = 20
                    Xy = np.empty((len(X_train), n_features + 1), dtype=int)
                    Xy[:, 0] = y_train
                    Xy[:, 1:] = np.round(X_train * n_bins).astype(int)  # Discretization
                    column_names = ['y'] + [f'x{i}' for i in range(n_features)]
                    df = pd.DataFrame(data=Xy, index=None, columns=column_names)
                    feature_names = set(pymrmr.mRMR(df, 'MIQ', k))
                    idx = np.asarray([int(feature_name[1:]) for feature_name in feature_names])
                    scores = np.zeros(n_features, dtype=float)
                    scores[idx] = 1
                    feature_names = set(pymrmr.mRMR(df, 'MIQ', 2 * k))
                    if 'y' in feature_names:
                        feature_names.remove('y')
                    idx = np.asarray([int(feature_name[1:]) for feature_name in feature_names])
                    scores2 = np.zeros(n_features, dtype=float)
                    scores2[idx] = 1
                elif method_name == 'cae':
                    from concrete_autoencoder import ConcreteAutoencoderFeatureSelector
                    import keras
                    def nn(x):
                        x = keras.layers.GaussianNoise(0.05)(x)
                        x = keras.layers.Dense(64)(x)
                        x = keras.layers.Dropout(0.2)(x)
                        x = keras.layers.LeakyReLU(alpha=0.2)(x)
                        x = keras.layers.Dense(64)(x)
                        x = keras.layers.Dropout(0.2)(x)
                        x = keras.layers.LeakyReLU(alpha=0.2)(x)
                        x = keras.layers.Dense(1, activation='sigmoid')(x)
                        return x
                    selector = ConcreteAutoencoderFeatureSelector(
                        K=k, output_function=nn, start_temp=10, min_temp=0.01, num_epochs=1000,
                        learning_rate=0.005, tryout_limit=1)
                    selector.fit(X_train, y_train)
                    y_hat = selector.get_params().predict(X_test)
                    indices = selector.get_support(indices=True).flatten()
                    scores = np.zeros(n_features, dtype=float)
                    scores[indices] = 1
                    selector = ConcreteAutoencoderFeatureSelector(
                        K=2*k, output_function=nn, start_temp=10, min_temp=0.01, num_epochs=1000,
                        learning_rate=0.005, tryout_limit=1)
                    selector.fit(X_train, y_train)
                    y_hat = selector.get_params().predict(X_test)
                    indices = selector.get_support(indices=True).flatten()
                    scores2 = np.zeros(n_features, dtype=float)
                    scores2[indices] = 1
                elif method_name == 'canceloutsigmoid':
                    wrapper = nn_wrapper.NNwrapper.create(dataset_name, len(X[0]), arch='cancelout-sigmoid')
                    wrapper.fit(X_train, y_train, epochs=300)
                    y_hat = wrapper.predict(X_test)
                    scores = wrapper.model.cancel_out.get_weights().data.numpy()
                    scores2 = scores
                elif method_name == 'canceloutsoftmax':
                    wrapper = nn_wrapper.NNwrapper.create(dataset_name, len(X[0]), arch='cancelout-softmax')
                    wrapper.fit(X_train, y_train, epochs=300)
                    y_hat = wrapper.predict(X_test)
                    scores = wrapper.model.cancel_out.get_weights().data.numpy()
                    scores2 = scores
                elif method_name == 'deeppink2o':
                    X_augmented = np.empty((X_train.shape[0], X_train.shape[1], 2))
                    X_augmented[:, :, 0] = X_train
                    X_augmented[:, :, 1] = knockoff_train
                    wrapper = nn_wrapper.NNwrapper.create(dataset_name, len(X_train[0]), arch='deeppink-2o')
                    wrapper.fit(X_augmented, y_train, epochs=300)
                    X_augmented = np.empty((X_test.shape[0], X_test.shape[1], 2))
                    X_augmented[:, :, 0] = X_test
                    X_augmented[:, :, 1] = knockoff_test
                    y_hat = wrapper.predict(X_augmented)
                    scores = wrapper.model.get_weights()
                    scores -= scores.min()
                    scores2 = scores
                elif method_name == 'deeppinku':

                    def shuffle_along_axis(M, axis=0):
                        idx = np.random.rand(*M.shape).argsort(axis=axis)
                        return np.take_along_axis(M, idx, axis=axis)

                    X_augmented = np.empty((X_train.shape[0], X_train.shape[1], 2))
                    X_augmented[:, :, 0] = X_train
                    # X_augmented[:, :, 1] = np.random.rand(X_train.shape[0], X_train.shape[1]) * 2 - 1
                    X_augmented[:, :, 1] = shuffle_along_axis(X_train, axis=0)
                    wrapper = nn_wrapper.NNwrapper.create(dataset_name, len(X_train[0]), arch='deeppink-2o')
                    wrapper.fit(X_augmented, y_train, epochs=300)
                    X_augmented = np.empty((X_test.shape[0], X_test.shape[1], 2))
                    X_augmented[:, :, 0] = X_test
                    # X_augmented[:, :, 1] = np.random.rand(X_test.shape[0], X_test.shape[1]) * 2 - 1
                    X_augmented[:, :, 1] = shuffle_along_axis(X_test, axis=0)
                    y_hat = wrapper.predict(X_augmented)
                    scores = wrapper.model.get_weights()
                    scores -= scores.min()
                    scores2 = scores
                else:
                    raise NotImplementedError(f'Unknown FS method "{method_name}"')

                if scores is not None:
                    assert scores.shape == (n_features,)
                    indices = np.argsort(np.abs(scores))
                    best_k.append(np.sum([i in correct_indices for i in indices[-k:]]) / k)
                    indices = np.argsort(np.abs(scores2))
                    best_2k.append(np.sum([i in correct_indices for i in indices[-2*k:]]) / k)
                if y_hat is not None:
                    aurocs.append(roc_auc_score(y_test, y_hat))
                    auprcs.append(average_precision_score(y_test, y_hat))

            if len(best_k) > 0:
                print(f'Average best-k: {np.mean(best_k)}')
                print(f'Average best-2k: {np.mean(best_2k)}')
            if len(aurocs) > 0:
                print(f'Average AUROC: {np.mean(aurocs)}')
                print(f'Average AUPRC: {np.mean(auprcs)}')
            if does_fs and has_classifier:
                f.write(f'{filename}\t{np.mean(best_k)}\t{np.mean(best_2k)}\t{np.mean(aurocs)}\t{np.mean(auprcs)}\n')
            elif does_fs:
                f.write(f'{filename}\t{np.mean(best_k)}\t{np.mean(best_2k)}\n')
            elif has_classifier:
                f.write(f'{filename}\t{np.mean(aurocs)}\t{np.mean(auprcs)}\n')
            else:
                raise NotImplementedError()


if __name__ == '__main__':

    method_names = [
        'nn', 'nnfs', 'rf', 'relief', 'fsnet', 'mrmr', 'lassonet', 'cae',
        'canceloutsigmoid', 'canceloutsoftmax', 'deeppink2o', 'deeppinku']
    parser = argparse.ArgumentParser()
    parser.add_argument('method', type=str, choices=method_names, help='Method name')
    args = parser.parse_args()

    process_dataset(args.method, 'ring+xor+sum', 6, [6, 8, 16, 32, 64, 128, 256, 512])
    process_dataset(args.method, 'ring+xor', 4, [4, 8, 16, 32, 64, 128, 256, 512])
    process_dataset(args.method, 'ring', 2, [2, 4, 8, 16, 32, 64, 128, 256, 512])
    process_dataset(args.method, 'xor', 2, [2, 4, 8, 16, 32, 64, 128, 256, 512])
