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

import os

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import pymrmr
import lassonet


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, 'data')
RESULTS_PATH = os.path.join(ROOT, 'results')


def load_indices(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    idx = []
    for line in lines:
        line = line.rstrip()
        if len(line) > 1:
            idx.append(int(float(line)))
    return np.asarray(idx)


def process_dataset(dataset_name, k, ns):
    with open(os.path.join(RESULTS_PATH, f'extra-{dataset_name}.txt'), 'w') as f:
        f.write('Dataset\tmRMR_bestK\tmRMR_best2K\tLassoNet_bestK\tLassoNet_best2K\n')
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

            best_k = []
            best_2k = []
            best_k_lassonet = []
            best_2k_lassonet = []
            for i in range(6):
                print(dataset_name, n_features, i)
                idx = load_indices(os.path.join(DATA_PATH, f'Indices_fold{i}.csv')).astype(int)
                mask = np.zeros(len(X), dtype=bool)
                mask[idx] = True

                # Run LassoNet
                model = lassonet.LassoNetClassifier(
                    hidden_dims=(32,),
                    n_iters=100,
                    batch_size=32,
                    dropout=(0.8 if (dataset_name != 'ring') else 0.2),
                    patience=10,
                    verbose=False)
                model.path(X[idx, :], y[idx])
                importances = np.abs(model.feature_importances_.numpy())
                indices = np.argsort(importances)
                best_k_lassonet.append(np.mean(indices[-k:] < k))
                best_2k_lassonet.append(np.sum(indices[-2*k:] < k) / k)

                # Run mRMR
                Xy = np.empty((len(X), n_features + 1), dtype=int)
                Xy[:, 0] = y
                Xy[:, 1:] = np.round(X * 20.).astype(int)  # Discretization
                column_names = ['y'] + [f'x{i}' for i in range(n_features)]
                df = pd.DataFrame(data=Xy[idx, :], index=None, columns=column_names)
                feature_names = pymrmr.mRMR(df, 'MIQ', k)
                assert 'y' not in feature_names
                idx = np.asarray([int(feature_name[1:]) for feature_name in feature_names])
                assert len(idx) == k
                best_k.append(np.mean(idx < k))
                feature_names = pymrmr.mRMR(df, 'MIQ', 2 * k)
                feature_names = set(feature_names)
                if 'y' in feature_names:
                    feature_names.remove('y')
                idx = np.asarray([int(feature_name[1:]) for feature_name in feature_names])
                best_2k.append(np.sum(idx < k) / k)

            print(f'Average best-k: {np.mean(best_k_lassonet)}')
            print(f'Average best-2k: {np.mean(best_2k_lassonet)}')

            f.write(f'{filename}\t{np.mean(best_k)}\t{np.mean(best_k)}\t{np.mean(best_k_lassonet)}\t{np.mean(best_2k_lassonet)}\n')


if __name__ == '__main__':

    process_dataset('ring+xor+sum', 6, [6, 8, 16, 32, 64, 128, 256, 512])
    process_dataset('ring+xor', 4, [4, 8, 16, 32, 64, 128, 256, 512])
    process_dataset('ring', 2, [2, 4, 8, 16, 32, 64, 128, 256, 512])
    process_dataset('xor', 2, [2, 4, 8, 16, 32, 64, 128, 256, 512])
