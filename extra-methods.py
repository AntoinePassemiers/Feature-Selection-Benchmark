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
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, 'data')


def load_array(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        line = line.rstrip()
        if len(line) > 0:
            data.append([float(x) for x in line.split(',')])
    return np.asarray(data)


k = 2
X = load_array(os.path.join(DATA_PATH, 'ring_1000samples-512feat.csv'))
y = X[:, -1].astype(int)
X = X[:, :-1]


print(X.shape)
best_k = []
best_2k = []
for i in range(6):
    idx = load_array(os.path.join(DATA_PATH, f'Indices_fold{i}.csv')).astype(int)
    mask = np.zeros(X.shape[0], dtype=bool)
    mask[idx] = 1
    X_train = X[~mask, :]
    y_train = y[~mask]

    importances = mutual_info_classif(X_train, y_train, n_neighbors=15)
    plt.plot(importances)

    idx = np.argsort(-importances)
    best_k.append(np.mean(idx[:k] < k))
    best_2k.append(2. * np.mean(idx[:2*k] < 2*k))
    print(best_k)
plt.show()

print(f'Average best-k: {np.mean(best_k)}')
print(f'Average best-2k: {np.mean(best_2k)}')
