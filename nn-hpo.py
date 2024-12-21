# -*- coding: utf-8 -*-
#
#  nn-hpo.py
#
#  Copyright 2024 Antoine Passemiers <antoine.passemiers@gmail.com>
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

import numpy as np
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import optuna

from src.dag import generate_dag_dataset
from src.data import generate_dataset
from src.nn_wrapper import NNwrapper, GaussianNoise, init_weights, Model


if __name__ == '__main__':

    all_best_params = []
    for dataset_name in ['xor', 'ring', 'ring+xor', 'ring+xor+sum']:
        X, _, y = generate_dataset(dataset_name, 500, 128)
        X = 2 * (X - 0.5)

        def objective(trial) -> float:

            aurocs, auprcs = [], []
            for i, (train_index, test_index) in enumerate(KFold(n_splits=6).split(X)):

                model = Model(
                    X.shape[1],
                    2,
                    latent_size=trial.suggest_int('latent_size', 2, 1000),
                    gaussian_noise=trial.suggest_float('gaussian_noise', 0, 1),
                    dropout=trial.suggest_float('dropout', 0, 1),
                    layer_norm=bool(trial.suggest_int('layer_norm', 0, 1)),
                    n_hidden_layers=trial.suggest_int('n_hidden_layers', 1, 5),
                    activation=trial.suggest_categorical('activation', ['relu', 'leakyrelu', 'prelu', 'tanh', 'sigmoid', 'mish', 'selu', 'hardswish'])
                )

                X_train, y_train = X[train_index, :], y[train_index]
                X_test, y_test = X[test_index, :], y[test_index]
                nn_wrapper = NNwrapper(model, 2)
                nn_wrapper.fit(
                    X_train,
                    y_train,
                    learning_rate=trial.suggest_float('learning_rate', 1e-7, 1e-2, log=True),
                    epochs=trial.suggest_int('max_epochs', 10, 1000),
                    batch_size=trial.suggest_int('batch_size', 4, 128),
                    weight_decay=trial.suggest_float('weight_decay', 1e-8, 1e-2, log=True),
                    early_stopping_patience=trial.suggest_int('early_stopping_patience', 1, 100),
                    optimizer=trial.suggest_categorical('optimizer', ['adam', 'sgd', 'adamw', 'rmsprop', 'adagrad']),
                    sam_type=trial.suggest_categorical('sam_type', ['sam', 'sam-adaptive', 'no-sam'])
                )
                y_hat = nn_wrapper.predict_proba(X_test)
                aurocs.append(roc_auc_score(y_test, y_hat))
                auprcs.append(average_precision_score(y_test, y_hat))

            return np.mean(aurocs), np.mean(auprcs)


        study = optuna.create_study(directions=['maximize', 'maximize'])
        study.optimize(objective, timeout=12*3600)

        trials = study.get_trials()

        data = []
        for trial in trials:
            data.append({
                'auroc': trial.values[0],
                'auprc': trial.values[1],
                'params': trial.params
            })
        with open(f'hps-{dataset_name}.json', 'w') as f:
            json.dump(data, f)
