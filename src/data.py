# -*- coding: utf-8 -*-
#
#  data.py
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

from typing import Tuple

import numpy as np


def _generate_points(n_samples: int, n_features: int) -> np.ndarray:
    return np.random.rand(n_samples, n_features)


def generate_ring_dataset(n_samples: int, n_features: int) -> Tuple[np.ndarray, np.ndarray]:
    X = _generate_points(n_samples, n_features)
    y = (np.abs(np.sqrt((X[:, 0] - 0.5) ** 2 + (X[:, 1] - 0.5) ** 2) - 0.35) <= 0.1151).astype(int)
    return X, y


def generate_xor_dataset(n_samples: int, n_features: int) -> Tuple[np.ndarray, np.ndarray]:
    X = _generate_points(n_samples, n_features)
    y = ((X[:, 0] - 0.5) * (0.5 - X[:, 1]) >= 0).astype(int)
    return X, y


def generate_ring_xor_dataset(n_samples: int, n_features: int) -> Tuple[np.ndarray, np.ndarray]:
    X = _generate_points(n_samples, n_features)
    cond1 = (np.abs(np.sqrt((X[:, 0] - 0.5) ** 2 + (X[:, 1] - 0.5) ** 2) - 0.35) <= 0.0704)
    cond2 = ((X[:, 2] - 0.5) * (0.5 - X[:, 3]) >= 0.0337)
    y = np.logical_or(cond1, cond2).astype(int)
    return X, y


def generate_ring_xor_sum_dataset(n_samples: int, n_features: int) -> Tuple[np.ndarray, np.ndarray]:
    X = _generate_points(n_samples, n_features)
    cond1 = (np.abs(np.sqrt((X[:, 0] - 0.5) ** 2 + (X[:, 1] - 0.5) ** 2) - 0.35) <= 0.0479)
    cond2 = ((X[:, 2] - 0.5) * (0.5 - X[:, 3]) >= 0.0598)
    cond3 = (X[:, 4] + X[:, 5] + np.random.normal(0, 0.2, size=n_samples) >= 1.4074)
    y = np.logical_or(np.logical_or(cond1, cond2), cond3).astype(int)
    return X, y


def generate_dataset(dataset_name: str, n_samples: int, n_features: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    name = dataset_name.lower().strip()
    if name == 'ring':
        X, y = generate_ring_dataset(n_samples, n_features)
    elif name == 'xor':
        X, y = generate_xor_dataset(n_samples, n_features)
    elif name == 'ring+xor':
        X, y = generate_ring_xor_dataset(n_samples, n_features)
    elif name == 'ring+xor+sum':
        X, y = generate_ring_xor_sum_dataset(n_samples, n_features)
    else:
        raise NotImplementedError(f'Unknown dataset "{dataset_name}"')

    # Generate knockoff features for DeepPINK
    X_tilde = np.random.uniform(-1, 1, X.shape)

    print(f'Class imbalance: {np.mean(y)}')

    return X, X_tilde, y
