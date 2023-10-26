# -*- coding: utf-8 -*-
#
#  fsnet.py
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
import torch

from src.utils import TrainingSet


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(1e-3)


class WeightPredictor(torch.nn.Module):

    def __init__(self, n_input, n_low, n_output, lhs=True, activation='none'):
        torch.nn.Module.__init__(self)
        self.U = None
        self.n_input = n_input
        self.n_low = n_low
        self.n_output = n_output
        self.lhs = lhs
        if self.lhs:
            shape = (self.n_input, self.n_low)
        else:
            shape = (self.n_low, self.n_output)
        self.weight = torch.nn.Parameter(torch.randn(*shape, requires_grad=True))
        if activation == 'tanh':
            self.activation = torch.nn.Tanh()
        else:
            self.activation = None
        torch.nn.init.xavier_uniform_(self.weight)

    def init(self, U):
        self.U = torch.FloatTensor(U)
        if self.lhs:
            assert self.U.size() == (self.n_low, self.n_output)
        else:
            assert self.U.size() == (self.n_input, self.n_low)

    def forward(self):
        assert self.U is not None
        if self.lhs:
            X = torch.mm(self.weight, self.U)
        else:
            X = torch.mm(self.U, self.weight)
        if self.activation is not None:
            X = self.activation(X)
        assert X.size() == (self.n_input, self.n_output)
        return X


class Selector(torch.nn.Module):

    def __init__(self, n_input, low, k, tau_0=10, tau_e=0.01):
        torch.nn.Module.__init__(self)
        self.n_input = n_input
        self.k = k
        self.tau_0 = tau_0
        self.tau_e = tau_e
        self.tau = self.tau_0
        self.weight_predictor = WeightPredictor(
            self.n_input, low, self.k, lhs=False, activation='none')
        # self.gumbel = torch.distributions.gumbel.Gumbel(0, 0.3)
        self.uniform = torch.distributions.uniform.Uniform(1e-5, 1. - 1e-5)

    def init(self, U):
        self.weight_predictor.init(U)

    def forward(self, X):
        logits = self.compute_logits()
        assert not np.any(np.isnan(logits.data.numpy()))

        # Concrete variables in the selection layer
        if self.training:
            g = self.sample_gumbel(logits.size())
            noisy_logits = (logits + g) / self.tau
        else:
            noisy_logits = logits
        M_T = torch.softmax(noisy_logits, 0)  # Array of shape (n_features, n_selected)

        assert not np.any(np.isnan(M_T.data.numpy()))

        # Select features
        if self.training:
            X_subset = torch.mm(X, M_T)
        else:
            indices, _ = Selector.uargmax(M_T)
            X_subset = X[:, indices]
        assert not np.any(np.isnan(X_subset.data.numpy()))
        return X_subset

    def sample_gumbel(self, shape):
        x = self.uniform.sample(shape)
        return -torch.log(-torch.log(x))

    def compute_logits(self):
        return self.weight_predictor()

    def update_temperature(self, e, n_epochs):
        self.tau = self.tau_0 * (self.tau_e / self.tau_0) ** ((e + 1) / n_epochs)

    def get_selected_features(self):
        logits = self.compute_logits()
        M_T = torch.softmax(logits, 0)
        idx, _ = Selector.uargmax(M_T)
        return idx.data.numpy()

    def get_feature_importances(self):
        logits = self.compute_logits()
        M_T = torch.softmax(logits, 0)
        #_, weights = Selector.uargmax(M_T)
        #return weights.data.numpy()
        return np.mean(M_T.data.numpy(), axis=1)

    @staticmethod
    def uargmax(A):
        A = A.data.numpy()

        # Ensure values are strictly positive
        A = A - A.min() + 1e-5

        indices = np.empty(A.shape[1], dtype=int)
        weights = np.zeros(A.shape[0], dtype=float)
        for k in range(A.shape[1]):
            i, j = np.unravel_index(np.argmax(A, axis=None), A.shape)
            indices[k] = i
            weights[i] = A[i, j]
            A[i, :] = 0
            A[:, j] = 0
        return torch.LongTensor(indices), torch.FloatTensor(weights)


class Encoder(torch.nn.Module):

    def __init__(self, n_input, n_output):
        torch.nn.Module.__init__(self)
        self.n_input = n_input
        self.n_output = n_output
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(self.n_input, self.n_output),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.apply(init_weights)

    def forward(self, X):
        # return self.layers(X)
        return X  # Identity function


class Decoder(torch.nn.Module):

    def __init__(self, n_input, n_output):
        torch.nn.Module.__init__(self)
        self.n_input = n_input
        self.n_output = n_output
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(self.n_input, self.n_output),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.apply(init_weights)

    def forward(self, X):
        # return self.layers(X)
        return X  # Identity function


class Reconstruction(torch.nn.Module):

    def __init__(self, n_input, n_low, n_output):
        torch.nn.Module.__init__(self)
        self.n_input = n_input
        self.n_output = n_output
        self.weight_predictor = WeightPredictor(
            self.n_input, n_low, self.n_output, lhs=True, activation='none')

    def init(self, U):
        self.weight_predictor.init(U)

    def forward(self, X):
        weights = self.weight_predictor()
        return torch.mm(X, weights)


class FSNet(torch.nn.Module):

    def __init__(self, model, n_input, n_bins, n_selected, n_classes):
        torch.nn.Module.__init__(self)
        self.model = model
        self.n_input = n_input
        self.n_bins = n_bins
        self.n_selected = n_selected
        self.n_classes = n_classes
        self.selector = Selector(n_input, n_bins, n_selected)
        self.encoder = Encoder(n_selected, n_selected)
        self.decoder = Decoder(n_selected, n_selected)
        self.reconstruction = Reconstruction(n_selected, n_bins, n_input)

    def fit(self, X, y, n_epochs=2000, batch_size=64, _lambda=10, weight_decay=1e-6):

        # Initialize weight predictors
        U = FSNet.compute_u(X, n_bins=self.n_bins)
        self.selector.init(U)
        self.reconstruction.init(U.t())

        dataset = TrainingSet(X, y)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, sampler=None, num_workers=0)
        self.model.train()
        if self.n_classes <= 2:
            criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        else:
            criterion = torch.nn.NLLLoss(reduction='mean')
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.9, patience=10, verbose=False, threshold=0.0001,
            threshold_mode='rel', cooldown=5, min_lr=1e-5, eps=1e-08)

        for e in range(n_epochs):

            # Lower the temperature
            self.selector.update_temperature(e, n_epochs)

            total_loss = 0
            for _X, _y in loader:

                optimizer.zero_grad()

                # Select a subset of features
                X_subset = self.selector.forward(_X)
                assert not np.any(np.isnan(X_subset.data.numpy()))

                # Predict the target variable from the selected subset of features
                X_latent = self.encoder.forward(X_subset)
                assert not np.any(np.isnan(X_latent.data.numpy()))
                y_hat = self.model.forward(X_latent)
                assert not np.any(np.isnan(y_hat.data.numpy()))

                # Reconstruct the input data
                X_reconstructed = self.decoder.forward(X_latent)
                assert not np.any(np.isnan(X_reconstructed.data.numpy()))
                X_reconstructed = self.reconstruction.forward(X_reconstructed)
                assert not np.any(np.isnan(X_reconstructed.data.numpy()))

                # Compute loss function
                if self.n_classes > 2:
                    loss1 = criterion(y_hat, _y)
                else:
                    loss1 = criterion(torch.squeeze(y_hat), torch.squeeze(_y))
                loss2 = _lambda * torch.mean((_X - X_reconstructed) ** 2)
                loss = loss1 + loss2

                # print(_X.size(), X_subset.size(), X_latent.size(), y_hat.size(), X_reconstructed.size())

                #print(loss1.item(), loss2.item())
                total_loss += loss.item()

                # Update parameters
                loss.backward()
                optimizer.step()
            scheduler.step(total_loss)

            # print(f'Total loss at epoch {e + 1}: {total_loss}')

        self.model.eval()

    def predict(self, X):
        X = torch.FloatTensor(X)
        X = self.selector.forward(X)
        X = self.encoder.forward(X)
        y_pred = self.model.forward(X)
        if self.n_classes <= 2:
            y_pred = torch.sigmoid(y_pred)
        else:
            y_pred = torch.softmax(y_pred, dim=1)
        return torch.squeeze(y_pred).data.numpy()

    def get_selected_features(self):
        return self.selector.get_selected_features()

    def get_feature_importances(self):
        importances = self.selector.get_feature_importances()
        return importances

    @staticmethod
    def compute_u(X, n_bins=20):
        n_features = X.shape[1]
        U = np.zeros((n_features, n_bins),dtype=float)
        for j in range(0, n_features):
            hist = np.histogram(X[:, j], n_bins)
            U[j, :] = 0.5 * hist[0][:] * (hist[1][:-1] + hist[1][1:])
        U -= U.mean()
        U /= U.std()
        return torch.FloatTensor(U)
