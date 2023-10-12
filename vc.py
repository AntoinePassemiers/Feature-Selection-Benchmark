# -*- coding: utf-8 -*-
#
#  vc.py
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

import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
import torch

from src.nn_wrapper import Model, NNwrapper
from src.fsnet import FSNet
from src.data import generate_dataset


xs    = [    2,     4,     8,    16,    32,    64,   128,   256,   512,  1024,  2048]

ys_nn = [0.500, 0.500, 0.563, 0.568, 0.854, 0.998,   1.0,   1.0,   1.0,   1.0,   1.0]
ys_si = [0.500, 0.500, 0.505, 0.513, 0.584, 0.938,   1.0,   1.0,   1.0,   1.0,   1.0]
ys_so = [0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500]
ys_dp = [0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500]
ys_ls = [0.500, 0.500, 0.537, 0.655, 0.639, 0.500, 0.525, 0.500, 0.500, 0.500, 0.565]
ys_fs = [0.567, 0.662, 0.563, 0.546, 0.520, 0.517, 0.500, 0.500, 0.500, 0.500, 0.500]

ys_ln = [0.500, 0.523, 0.542, 0.569, 0.614, 0.999,   1.0,   1.0,   1.0,   1.0,   1.0]
ys_lc = [0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.701, 0.778, 0.876, 0.897, 0.909]

ts_nn = [0.988, 0.996, 0.984, 0.981, 0.994,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0]
ts_si = [0.984, 0.991, 0.988, 0.983, 0.989, 0.994,   1.0,   1.0,   1.0,   1.0,   1.0]
ts_so = [0.972, 0.985, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500]
ts_dp = [0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500]
ts_ls = [0.998, 0.996, 0.999, 0.998, 0.997,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0]
ts_fs = [0.992, 0.988, 0.989, 0.988, 0.546, 0.513, 0.519, 0.500, 0.500, 0.500, 0.500]



"""
plt.figure(figsize=(16, 10))

ax = plt.subplot(3, 2, 1)
plt.title('NN')
#plt.plot(xs, ys_si, marker='o', label='CancelOut (sigmoid)', color='orangered')
#plt.plot(xs, ys_so, marker='o', label='CancelOut (softmax)', color='orange')
#plt.plot(xs, ys_dp, marker='x', label='DeepPINK', color='pink')
#plt.plot(xs, ys_ln, marker='*', label='LassoNet', color='chocolate')
plt.plot(xs, ys_nn, marker='x', label='Decoys only', color='grey')
plt.plot(xs, ts_nn, marker='x', label='XOR + decoys', color='black')
plt.xscale('log')
plt.legend()
ax.spines[['right', 'top']].set_visible(False)
plt.xlabel('Number of features')
plt.ylabel('Training accuracy')
ax.set_xticks(xs)
ax.set_xticklabels([str(n_features) for n_features in xs])
plt.grid(alpha=0.4, linestyle='--', linewidth=0.5, color='grey')
prop = {'family': 'Century gothic', 'size': 9}
plt.legend(prop=prop, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, frameon=False)

ax = plt.subplot(3, 2, 2)
plt.title('LassoNet')
plt.plot(xs, ys_ls, marker='*', label='Decoys only', color='chocolate')
plt.plot(xs, ts_ls, marker='x', label='XOR + decoys', color='black')
plt.xscale('log')
plt.legend()
ax.spines[['right', 'top']].set_visible(False)
plt.xlabel('Number of features')
plt.ylabel('Training accuracy')
ax.set_xticks(xs)
ax.set_xticklabels([str(n_features) for n_features in xs])
plt.grid(alpha=0.4, linestyle='--', linewidth=0.5, color='grey')
plt.legend(prop=prop, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, frameon=False)

ax = plt.subplot(3, 2, 3)
plt.title('CancelOut (sigmoid)')
plt.plot(xs, ys_si, marker='o', label='Decoys only', color='orangered')
plt.plot(xs, ts_si, marker='x', label='XOR + decoys', color='black')
plt.xscale('log')
plt.legend()
ax.spines[['right', 'top']].set_visible(False)
plt.xlabel('Number of features')
plt.ylabel('Training accuracy')
ax.set_xticks(xs)
ax.set_xticklabels([str(n_features) for n_features in xs])
plt.grid(alpha=0.4, linestyle='--', linewidth=0.5, color='grey')
plt.legend(prop=prop, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, frameon=False)

ax = plt.subplot(3, 2, 4)
plt.title('CancelOut (softmax)')
plt.plot(xs, ys_so, marker='o', label='Decoys only', color='orange')
plt.plot(xs, ys_ln, linestyle='--', marker='o', label='Decoys only (LayerNorm)', color='orangered')
plt.plot(xs, ts_so, marker='x', label='XOR + decoys', color='black')
plt.xscale('log')
plt.legend()
ax.spines[['right', 'top']].set_visible(False)
plt.xlabel('Number of features')
plt.ylabel('Training accuracy')
ax.set_xticks(xs)
ax.set_xticklabels([str(n_features) for n_features in xs])
plt.grid(alpha=0.4, linestyle='--', linewidth=0.5, color='grey')
plt.legend(prop=prop, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, frameon=False)

ax = plt.subplot(3, 2, 5)
plt.title('DeepPINK')
plt.plot(xs, ys_dp, marker='x', label='Decoys only', color='pink')
plt.plot(xs, ys_lc, linestyle='--', marker='x', label='Decoys only (1 LC layer)', color='deeppink')
plt.plot(xs, ts_dp, marker='x', label='XOR + decoys', color='black')
plt.plot([0], [1], alpha=0)
plt.xscale('log')
plt.legend()
ax.spines[['right', 'top']].set_visible(False)
plt.xlabel('Number of features')
plt.ylabel('Training accuracy')
ax.set_xticks(xs)
ax.set_xticklabels([str(n_features) for n_features in xs])
plt.grid(alpha=0.4, linestyle='--', linewidth=0.5, color='grey')
plt.legend(prop=prop, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, frameon=False)

ax = plt.subplot(3, 2, 6)
plt.title('FSNet')
plt.plot(xs, ys_fs, marker='s', label='Decoys only', color='navy')
plt.plot(xs, ts_fs, marker='x', label='XOR + decoys', color='black')
plt.plot([0], [1], alpha=0)
plt.xscale('log')
plt.legend()
ax.spines[['right', 'top']].set_visible(False)
plt.xlabel('Number of features')
plt.ylabel('Training accuracy')
ax.set_xticks(xs)
ax.set_xticklabels([str(n_features) for n_features in xs])
plt.grid(alpha=0.4, linestyle='--', linewidth=0.5, color='grey')
plt.legend(prop=prop, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, frameon=False)

plt.tight_layout()

plt.savefig('training.png', dpi=300)
import sys; sys.exit(0)
"""



k = 32

#X = np.random.rand(1000, k) * 2 - 1
#X_tilde = np.random.rand(1000, k) * 2 - 1
#y = np.zeros(1000, dtype=int)
#y[:500] = 1
X, X_tilde, y = generate_dataset('xor', 1000, k)
X = X * 2 - 1
X_tilde = X_tilde * 2 - 1

X_augmented = np.empty((X.shape[0], X.shape[1], 2))
X_augmented[:, :, 0] = X
X_augmented[:, :, 1] = X_tilde



method_name = 'fsnet'

if method_name == 'lassonet':
    import lassonet
    model = lassonet.LassoNetClassifier(
        hidden_dims=(32, 32),
        n_iters=(30, 30),
        batch_size=64,
        dropout=0,
        patience=10,
        lambda_start=3.2768,
        tol=0.9999,
        verbose=False)
    path = model.path(X, y)
    val_losses = np.asarray([save.val_loss for save in path])
    state_dict = path[np.argmin(val_losses)].state_dict
    model.load(state_dict)
    y_pred = model.predict(X)
elif method_name == 'nn':
    model = NNwrapper(Model(X.shape[1], 2), 2)
    model.fit(X, y, val=0)
    y_pred = model.predict(X)
    y_pred = (y_pred > 0.5).astype(int)
elif method_name == 'cancelout-sigmoid':
    model = NNwrapper.create('random', X.shape[1], 2, arch='cancelout-sigmoid')
    model.fit(X, y, val=0)
    y_pred = model.predict(X)
    y_pred = (y_pred > 0.5).astype(int)
elif method_name == 'cancelout-softmax':
    model = NNwrapper.create('random', X.shape[1], 2, arch='cancelout-softmax')
    model.fit(X, y, val=0)
    y_pred = model.predict(X)
    y_pred = (y_pred > 0.5).astype(int)
elif method_name == 'deeppink':
    model = NNwrapper.create('random', X.shape[1], 2, arch='deeppink')
    model.fit(X_augmented, y)
    y_pred = model.predict(X_augmented)
    y_pred = (y_pred > 0.5).astype(int)
elif method_name == 'fsnet':
    n_selected = 16  #min(2 * 2, k)
    fsnet = FSNet(Model(n_selected, 2), k, 30, n_selected, 2)
    fsnet.fit(X, y)
    y_pred = fsnet.predict(X)
    y_pred = (y_pred > 0).astype(int)


print(np.mean(y == y_pred))
