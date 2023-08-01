# -*- coding: utf-8 -*-
#
#  nn-hpo.py
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

import numpy as np
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neural_network import MLPClassifier

from src.dag import generate_dag_dataset
from src.data import generate_dataset
from src.nn_wrapper import NNwrapper


if __name__ == '__main__':

    param_grid = {
        'activation': ['logistic', 'tanh', 'relu'],
        'hidden_layer_sizes': [
            [100],
            [100, 50],
            [200, 100, 50],
            [50, 20],
            [50],
            [20],
            [10],
            [200, 200, 200],
            [200, 200, 200, 200],
            [200, 200, 200, 200, 200]
        ],
        'solver': ['sgd', 'adam'],
        'alpha': [0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'learning_rate_init': [0.00001, 0.0001, 0.001, 0.01, 0.1],
        'max_iter': [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    }

    all_best_params = []
    for dataset_name in ['xor', 'ring', 'ring+xor', 'ring+xor+sum']:
        X, y = generate_dataset(dataset_name, 500, 2048)

        model = MLPClassifier()
        clf = GridSearchCV(model, param_grid=param_grid, cv=KFold(n_splits=6))
        clf.fit(X, y)
        all_best_params.append(clf.best_params_)
        print(f'Best parameters for dataset {dataset_name}: {clf.best_params_}')
    for params in all_best_params:
        print(params)
