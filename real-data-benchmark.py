# -*- coding: utf-8 -*-
#
#  real-data-benchmark.py
#
#  Copyright 2023 Antoine Passemiers <antoine.passemiers@gmail.com>
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

import json
import os
import PIL.Image

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import mnist

from src.core import run_fs_method
from src.knockoff import generate_gaussian_knockoffs


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, 'data')
RESULTS_PATH = os.path.join(ROOT, 'results',  'external-data')
if not os.path.isdir(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

METHOD_NAMES = [
    # 'fsnet',
    'lassonet',
    #'nn', 'rf', 'canceloutsigmoid', 'canceloutsoftmax',
]

for method_name in METHOD_NAMES:

    # for DATASET in ['fashion', 'isolet', 'mice', 'har', 'mnist', 'coil20']:
    for DATASET in ['fashion']:

        if os.path.exists(os.path.join(RESULTS_PATH, f'{DATASET}-{method_name}.json')):
            continue

        print(f'Running method "{method_name}" on dataset "{DATASET}"')

        if DATASET == 'isolet':
            data = np.loadtxt(os.path.join(DATA_PATH, 'isolet', 'isolet1+2+3+4.data'), delimiter=',')
            X_train = data[:, :-1]
            y_train = data[:, -1].astype(int) - 1
            data = np.loadtxt(os.path.join(DATA_PATH, 'isolet', 'isolet5.data'), delimiter=',')
            X_test = data[:, :-1]
            y_test = data[:, -1].astype(int) - 1
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        elif DATASET == 'mice':
            df = pd.read_excel(os.path.join(DATA_PATH, 'mice+protein+expression', 'Data_Cortex_Nuclear.xls'))
            df.drop(columns=['MouseID', 'Genotype', 'Treatment', 'Behavior'], inplace=True)
            y = LabelEncoder().fit_transform(df['class'].to_numpy())
            df.drop(columns=['class'], inplace=True)
            X = df.to_numpy()
            for j in range(X.shape[1]):
                mask = np.isnan(X[:, j])
                X[mask, j] = np.nanmedian(X[:, j])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        elif DATASET == 'har':
            X_train = np.loadtxt(os.path.join(DATA_PATH, 'UCI HAR Dataset', 'train', 'X_train.txt'))
            y_train = np.loadtxt(os.path.join(DATA_PATH, 'UCI HAR Dataset', 'train', 'y_train.txt'))
            X_test = np.loadtxt(os.path.join(DATA_PATH, 'UCI HAR Dataset', 'test', 'X_test.txt'))
            y_test = np.loadtxt(os.path.join(DATA_PATH, 'UCI HAR Dataset', 'test', 'y_test.txt'))
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            y_train = y_train.astype(int) - 1
            y_test = y_test.astype(int) - 1
        elif DATASET in {'fashion', 'mnist'}:
            mndata = mnist.MNIST(os.path.join(DATA_PATH, DATASET))
            X_train, y_train = mndata.load_training()
            X_test, y_test = mndata.load_testing()
            X_train, y_train = np.asarray(X_train), np.asarray(y_train)
            X_test, y_test = np.asarray(X_test), np.asarray(y_test)
            X_train = X_train.astype(float) / 255
            X_test = X_test.astype(float) / 255
        elif DATASET == 'coil20':
            X, y = [], []
            for filename in os.listdir(os.path.join(DATA_PATH, 'coil-20-proc')):
                filepath = os.path.join(DATA_PATH, 'coil-20-proc', filename)
                label = int(filename.split('__')[0].replace('obj', '')) - 1
                image = np.asarray(PIL.Image.open(filepath).convert('L'))
                X.append(image.flatten())
                y.append(label)
            X = np.asarray(X)
            y = np.asarray(y)
            X = X.astype(float) / 255
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        else:
            raise NotImplementedError(f'Unknown dataset "{DATASET}"')
        assert not np.any(np.isnan(X_train))
        assert not np.any(np.isnan(X_test))
        assert len(y_train.shape) == 1
        assert len(y_test.shape) == 1

        X_train = X_train[:, :10]
        X_test = X_test[:, :10]  # TODO

        if method_name == 'deeppink':
            ko_filepath = os.path.join(DATA_PATH, f'knockoff-{DATASET}.npy')
            if not os.path.exists(ko_filepath):
                X_tilde = generate_gaussian_knockoffs(np.concatenate((X_train, X_test), axis=0))
                np.save(ko_filepath, X_tilde)
            else:
                X_tilde = np.load(ko_filepath)
            X_tilde_train = X_tilde[:len(X_train)]
            X_tilde_test = X_tilde[len(X_train):]
        else:
            X_tilde_train = X_train
            X_tilde_test = X_test

        k = int(0.25 * X_train.shape[1])

        y_train_hat, y_hat, _, _ = run_fs_method(
            DATASET,
            method_name,
            X_train,
            X_tilde_train,
            y_train,
            X_test,
            X_tilde_test,
            k
        )

        results = {
            'auroc': roc_auc_score(y_test, y_hat, multi_class='ovr'),
            'aupr': np.mean([average_precision_score(y_test == k, y_hat[:, k], average='macro') for k in range(y_hat.shape[1])])
        }
        with open(os.path.join(RESULTS_PATH, f'{DATASET}-{method_name}.json'), 'w') as f:
            json.dump(results, f)
