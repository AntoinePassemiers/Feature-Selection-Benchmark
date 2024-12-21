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

from typing import Tuple
import time
import tracemalloc
import random
import json
import os
import PIL.Image

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import mnist

from src.core import run_fs_method, run_fa_method
from src.nn_wrapper import NNwrapper, Model
from src.knockoff import generate_gaussian_knockoffs


SEED = 0xCAFE
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, 'data')
RESULTS_PATH = os.path.join(ROOT, 'results',  'external-data')
if not os.path.isdir(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

METHOD_NAMES = [
    'random',
    'Saliency',
    'InputXGradient',
    'IG_noMul',
    'SmoothGrad',
    'GuidedBackprop',
    'DeepLift',
    'Deconvolution',
    'FeatureAblation',
    'FeaturePermutation',
    'ShapleyValueSampling',
    'canceloutsigmoid',
    'canceloutsoftmax',
    'deeppink',
    'rf',
    'treeshap',
    'relief',
    'lassonet',
    'mi',
    'mrmr',
    'fsnet',
    'cae',
]

#DATASETS = ['arcene', 'madelon', 'dexter', 'dorothea', 'gisette', 'fashion', 'isolet', 'mice', 'har', 'mnist', 'coil20']
DATASETS = ['dexter']


def load_nips2003_labels(filepath: str) -> np.ndarray:
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.rstrip()
            if len(line) > 0:
                data.append(int(line))
    y = np.asarray(data, dtype=int)
    return (y > 0).astype(int)


def load_nips2003_dense_matrix(filepath: str) -> np.ndarray:
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            elements = line.rstrip().split(' ')
            if len(elements) > 0:
                data.append([int(x) for x in elements])
    return np.asarray(data, dtype=int)


def load_nips2003_sparse_matrix(filepath: str, m: int) -> np.ndarray:
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            elements = line.rstrip().split(' ')
            if len(elements) > 0:
                xs = np.zeros(m, dtype=int)
                for el in elements:
                    j, value = el.split(':')
                    j, value = int(j) - 1, int(value)
                    assert 0 <= j < m
                    xs[j] = value
                data.append(xs)
    return np.asarray(data, dtype=int)
    return (y > 0).astype(int)


def load_nips2003_binary_sparse_matrix(filepath: str, m: int) -> np.ndarray:
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            elements = line.rstrip().split(' ')
            if len(elements) > 0:
                xs = np.zeros(m, dtype=int)
                for j in elements:
                    j = int(j) - 1
                    assert 0 <= j < m
                    xs[j] = 1
                data.append(xs)
    return np.asarray(data, dtype=int)


def load_nips2003_dataset(dataset_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    folder = os.path.join(DATA_PATH, dataset_name)
    sub_folder = os.path.join(DATA_PATH, dataset_name, dataset_name.upper())
    if dataset_name in {'arcene', 'gisette', 'madelon'}:
        X_train = load_nips2003_dense_matrix(os.path.join(sub_folder, f'{dataset_name}_train.data'))
        X_test = load_nips2003_dense_matrix(os.path.join(sub_folder, f'{dataset_name}_valid.data'))
    elif dataset_name == 'dexter':
        X_train = load_nips2003_sparse_matrix(os.path.join(sub_folder, f'{dataset_name}_train.data'), 20000)
        X_test = load_nips2003_sparse_matrix(os.path.join(sub_folder, f'{dataset_name}_valid.data'), 20000)
    elif dataset_name == 'dorothea':
        X_train = load_nips2003_binary_sparse_matrix(os.path.join(sub_folder, f'{dataset_name}_train.data'), 100000)
        X_test = load_nips2003_binary_sparse_matrix(os.path.join(sub_folder, f'{dataset_name}_valid.data'), 100000)
    else:
        raise NotImplementedError(f'Unknown dataset "{dataset_name}"')
    y_train = load_nips2003_labels(os.path.join(sub_folder, f'{dataset_name}_train.labels'))
    y_test = load_nips2003_labels(os.path.join(folder, f'{dataset_name}_valid.labels'))
    return X_train, y_train, X_test, y_test


random.shuffle(DATASETS)
for DATASET in DATASETS:

    if DATASET in {'arcene', 'dexter', 'dorothea', 'gisette', 'madelon'}:
        X_train, y_train, X_test, y_test = load_nips2003_dataset(DATASET)
    elif DATASET == 'isolet':
        data = np.loadtxt(os.path.join(DATA_PATH, 'isolet', 'isolet1+2+3+4.data'), delimiter=',')
        X_train = data[:, :-1]
        y_train = data[:, -1].astype(int) - 1
        data = np.loadtxt(os.path.join(DATA_PATH, 'isolet', 'isolet5.data'), delimiter=',')
        X_test = data[:, :-1]
        y_test = data[:, -1].astype(int) - 1
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
    elif DATASET == 'har':
        X_train = np.loadtxt(os.path.join(DATA_PATH, 'UCI HAR Dataset', 'train', 'X_train.txt'))
        y_train = np.loadtxt(os.path.join(DATA_PATH, 'UCI HAR Dataset', 'train', 'y_train.txt'))
        X_test = np.loadtxt(os.path.join(DATA_PATH, 'UCI HAR Dataset', 'test', 'X_test.txt'))
        y_test = np.loadtxt(os.path.join(DATA_PATH, 'UCI HAR Dataset', 'test', 'y_test.txt'))
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

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n_classes = int(np.max(y_train)) + 1
    n_features = X_train.shape[1]
    nn_wrapper = None

    decoy_fraction = 0.5
    if DATASET in {'arcene', 'gisette'}:
        decoy_fraction = 0.3
    elif DATASET == 'madelon':
        decoy_fraction = 0.96
    k = int(round((1. - decoy_fraction) * X_train.shape[1]))

    random.shuffle(METHOD_NAMES)
    for method_name in METHOD_NAMES:

        #if os.path.exists(os.path.join(RESULTS_PATH, f'{DATASET}-{method_name}.json')):
        #    continue

        print(f'Running method "{method_name}" on dataset "{DATASET}"')

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

        t0 = time.time()
        tracemalloc.start()

        if method_name in {'IG_noMul', 'Saliency', 'DeepLift', 'InputXGradient', 'SmoothGrad', 'GuidedBackprop', 'Deconvolution', 'FeatureAblation', 'FeaturePermutation', 'ShapleyValueSampling'}:
            if nn_wrapper is None:
                nn_wrapper = NNwrapper.create(DATASET, n_features, n_classes)
                nn_wrapper.fit(X_train, y_train)
            scores = run_fa_method(nn_wrapper, X_test, method_name)
        else:
            try:
                _, _, scores, _ = run_fs_method(
                    DATASET,
                    method_name,
                    X_train,
                    X_tilde_train,
                    y_train,
                    X_test,
                    X_tilde_test,
                    k,
                    _2k=False
                )
            except:
                continue
        _, max_memory_usage = tracemalloc.get_traced_memory()
        runtime = time.time() - t0

        assert scores is not None
        idx = np.argsort(scores)[-k:]

        model = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=SEED)
        model.fit(X_train[:, idx], y_train)
        y_hat = model.predict_proba(X_test[:, idx])

        if (y_hat.shape[1] == 2):
            y_hat = y_hat[:, 1]

        if len(y_hat.shape) > 1:
            results = {
                'auroc': roc_auc_score(y_test, y_hat, multi_class='ovr'),
                'aupr': np.mean([average_precision_score(y_test == k, y_hat[:, k], average='macro') for k in range(y_hat.shape[1])]),
                'y': [int(x) for x in y_test],
                'y-hat': [[float(x) for x in xs] for xs in y_hat],
            }
        else:
            results = {
                'auroc': roc_auc_score(y_test, y_hat),
                'aupr': average_precision_score(y_test, y_hat),
                'y': [int(x) for x in y_test],
                'y-hat': [float(x) for x in y_hat],
            }
        results['computation-time'] = runtime
        results['max-memory-usage'] = max_memory_usage
        print('Perf', results['auroc'], results['aupr'])
        with open(os.path.join(RESULTS_PATH, f'{DATASET}-{method_name}.json'), 'w') as f:
            json.dump(results, f)
