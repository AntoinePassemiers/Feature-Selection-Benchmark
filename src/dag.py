# -*- coding: utf-8 -*-
#
#  dag.py
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
from typing import Optional, List, Tuple

import numpy as np
import scipy.special

from src.knockoff import generate_gaussian_knockoffs


class Node:

    def __init__(self, sigma: float = 1.):
        self.data: Optional[np.ndarray] = None
        self.parents: List[Tuple[float, Node]] = []
        self.sigma: float = sigma

    def add(self, node: 'Node', weight: float):
        self.parents.append((weight, node))

    def reset(self):
        self.data = None

    def sample(self, n: int) -> np.ndarray:
        if self.data is not None:
            return self.data
        values = np.zeros(n, dtype=float)
        if len(self.parents) > 0:
            for weight, parent in self.parents:
                values += weight * parent.sample(n)
            values /= len(self.parents)
        values += np.random.normal(0, self.sigma, size=n)
        std_ = np.std(values)
        if std_ > 0:
            values = 2. * (values - np.mean(values)) / std_
        values = scipy.special.expit(values)
        self.data = values
        assert values.shape == (n,)
        return values


def random_dag(n_features=2000, n_relevant=20, n_irrelevant=1000, density=0.005):
    assert n_relevant + n_irrelevant + 1 <= n_features
    n = n_features + 1
    A = (np.random.rand(n, n) < density).astype(int)
    np.fill_diagonal(A, 0)

    # Ensure only (i, j) edges exist, such that i < j
    A[np.tril_indices(n)] = 0

    # Add irrelevant features
    idx = np.arange(n)
    np.random.shuffle(idx)
    idx = idx[:n_irrelevant]
    A[:, idx] = 0
    A[idx, :] = 0

    # Shuffle variables
    idx = np.arange(n)
    np.random.shuffle(idx)
    A = A[:, idx][idx, :]

    # Make sure the k first features (the most relevant ones)
    # have a 0 in-degree, and are all linked to the last feature (the target one)
    #A[:, :n_relevant] = 0
    #A[:n_relevant, -1] = 1
    #A[n_relevant:, -1] = 0

    #G = networkx.from_numpy_matrix(A, create_using=networkx.DiGraph())
    #assert networkx.is_directed_acyclic_graph(G)

    # Naive algorithm for finding all pairs of connected vertices.
    # The performance of this algorithm will strongly depend on the topology
    # of the input graph, but it proved to be quite efficient in practice.
    # Note: possible improvement, compute accessibility matrix (e.g. with Floyd-Warshall algorithm)
    C_old = np.copy(np.asarray(A, dtype=bool))
    while True:
        C_new = np.logical_or(C_old, np.dot(C_old.astype(int), C_old.astype(int)) > 0)
        if np.all(C_new == C_old):
            break
        C_old = C_new

    # Find forks
    F = (np.dot(C_new.T.astype(int), C_new.astype(int)) > 0).astype(bool)

    all_indices = set(list(range(len(A) - 1)))
    chain_indices = set(list(np.where(C_new[:-1, -1])[0]))
    fork_indices = set(list(np.where(F[:-1, -1])[0])) - chain_indices
    remaining_indices = all_indices - chain_indices - fork_indices
    idx = np.asarray(list(chain_indices) + list(fork_indices) + list(remaining_indices) + [len(A) - 1])
    assert len(idx) == len(A)
    A = A[:, idx][idx, :]

    k = len(chain_indices)
    k2 = len(fork_indices) + k

    return A, k, k2


def generate_dag_dataset(
        n_samples,
        n_features,
        min_n_relevant,
        min_n_irrelevant,
        density=0.004,
        sigma=0.2
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    A, k, k2 = random_dag(n_features, n_relevant=min_n_relevant, n_irrelevant=min_n_irrelevant, density=density)
    nodes = [Node(sigma=sigma) for _ in range(len(A))]
    for i, j in zip(*np.where(A)):
        weight = float(np.random.uniform(-1, 1))
        nodes[j].add(nodes[i], weight)
    X = np.empty((n_samples, n_features), dtype=float)
    for i in range(n_features):
        X[:, i] = nodes[i].sample(n_samples)
    y_continuous = nodes[-1].sample(n_samples)
    y = (y_continuous > np.median(y_continuous)).astype(int)

    print(f'k: {k}, {k2}, {n_features}')
    assert X.shape == (n_samples, n_features)

    return X, y, k, k2


def load_dag_dataset(folder):
    filepath = os.path.join(folder, 'dag.npz')
    if not os.path.exists(filepath):
        X, y, k, k2 = generate_dag_dataset(1000, 2000, 20, 1000, density=0.004, sigma=0.2)
        X_tilde = generate_gaussian_knockoffs(X)
        np.savez(filepath, X=X, X_tilde=X_tilde, y=y, k=int(k), k2=int(k2))
    else:
        data = np.load(filepath)
        X = data['X']
        X_tilde = data['X_tilde']
        y = data['y']
        k = int(data['k'])
        k2 = int(data['k2'])

    return X, X_tilde, y, k, k2
