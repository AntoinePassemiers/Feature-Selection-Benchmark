# -*- coding: utf-8 -*-
#
#  knockoff.py
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
import scipy.linalg
import torch


def make_g_spd(sigma):
    sigma = torch.FloatTensor(sigma)
    s = torch.nn.Parameter(0.0001 * torch.ones(len(sigma)))
    optimizer = torch.optim.SGD([s], lr=1e-3)
    for _ in range(100):
        optimizer.zero_grad()
        G1 = torch.cat([sigma, sigma - torch.diag(s)], dim=1)
        G2 = torch.cat([sigma - torch.diag(s), sigma], dim=1)
        G = torch.cat([G1, G2], dim=0)
        eigenvalues = torch.linalg.eigvalsh(G)
        loss = -torch.min(eigenvalues)
        loss.backward()
        print(loss.item())
        optimizer.step()
    return s.cpu().data.numpy()


def generate_gaussian_knockoffs(X, eps=1e-3):
    mu = np.mean(X, axis=0)
    p = len(mu)
    sigma = np.cov(X, rowvar=False)

    # Regularization
    lambda_ = 0.8
    sigma = lambda_ * np.diag(np.diagonal(sigma)) + (1. - lambda_) * sigma

    s = np.diagonal(sigma)

    # Ensure G is spd
    G = np.block([
        [sigma, sigma - np.diag(s)],
        [sigma - np.diag(s), sigma]
    ])
    # assert G.shape[0] == np.linalg.matrix_rank(G)

    S = np.diag(s)
    sigma_inv_d = scipy.linalg.lstsq(sigma, S)[0]
    V = 2. * S - np.dot(S, sigma_inv_d)
    L = np.linalg.cholesky(V + eps * np.eye(p))

    n, p = X.shape
    mu_tilde = X - np.dot(X - np.tile(mu, (n, 1)), sigma_inv_d)
    return mu_tilde + np.dot(np.random.normal(size=mu_tilde.shape), L.T)
