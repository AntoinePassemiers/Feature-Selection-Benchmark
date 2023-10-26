# -*- coding: utf-8 -*-
#
#  curse.py
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
import scipy.spatial
import scipy.stats

from src.data import generate_dataset


ks = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

plt.figure(figsize=(16, 8))

scores = []
for i, k in enumerate(ks):

    ax = plt.subplot(1, len(ks), i + 1)
    if i == 0:
        plt.ylabel('Pairwise distances')

    X, X_tilde, y = generate_dataset('xor', 1000, k)
    X = X * 2 - 1

    D = scipy.spatial.distance.cdist(X, X)
    mask = (y[:, np.newaxis] == y[np.newaxis, :])
    D_same = D[mask]
    D_same = D_same[D_same > 0]
    D_diff = D[~mask]

    p_value = scipy.stats.ks_2samp(D_same, D_diff).pvalue
    print(p_value)

    plt.title(f'p={p_value:.3f}')
    plt.hist(D_same, alpha=0.4, bins=200, density=True, orientation='horizontal', label='Same class', color='darkcyan')
    plt.hist(D_diff, alpha=0.4, bins=200, density=True, orientation='horizontal', label='Different class', color='tomato')
    ax.spines[['right', 'top']].set_visible(False)
    plt.xticks([0.5], [f'm = {k}'])
    plt.grid(alpha=0.4, linestyle='--', linewidth=0.5, color='darkcyan')

    if i == len(ks) - 1:
        prop = {'family': 'Century gothic', 'size': 9}
        plt.legend(prop=prop, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, frameon=False)

plt.savefig('distances.png', dpi=300)
