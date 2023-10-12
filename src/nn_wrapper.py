# -*- coding: utf-8 -*-
#
#  nn_wrapper.py
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

import captum.attr
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.cancelout import CancelOut
from src.deeppink import DeepPINK
from src.utils import TrainingSet, TestSet


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        if m.weight.size()[1] == 1:
            torch.nn.init.xavier_uniform_(m.weight)
        else:
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
        m.bias.data.fill_(1e-3)


class GaussianNoise(torch.nn.Module):

    def __init__(self, stddev):
        torch.nn.Module.__init__(self)
        self.stddev = stddev

    def forward(self, X):
        if self.training:
            X = X + self.stddev * torch.randn(*X.size())
        return X


class Model(torch.nn.Module):

    def __init__(self, input_size, n_classes, latent_size=16):
        torch.nn.Module.__init__(self)
        n_out = 1 if (n_classes <= 2) else n_classes
        self.layers = torch.nn.Sequential(
            # torch.nn.LayerNorm(input_size),
            GaussianNoise(0.1),
            torch.nn.Linear(input_size, latent_size),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(latent_size, latent_size),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(latent_size, n_out))
        self.apply(init_weights)
            
    def forward(self, x):
        return self.layers(x)


class ModelWithCancelOut(torch.nn.Module):

    def __init__(self, input_size, n_classes, latent_size=16, cancel_out_activation='sigmoid'):
        torch.nn.Module.__init__(self)
        self.cancel_out = CancelOut(input_size, activation=cancel_out_activation)
        self.model = Model(input_size, n_classes, latent_size=latent_size)

    def forward(self, x):
        x = self.cancel_out(x)
        return self.model(x)


class NNwrapper:

    def __init__(self, model, n_classes):
        self.model = model
        self.n_classes = n_classes
        self.loss_callbacks = []
        self.trained = False

    def add_loss_callback(self, func):
        self.loss_callbacks.append(func)

    def fit(self, X, Y, device='cpu', learning_rate=0.0005, epochs=1000, batch_size=64, weight_decay=1e-2, val=0.2):

        if val > 0:
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=val)
        else:
            X_train, y_train = X, Y
            X_test = np.asarray([])
            y_test = np.asarray([])

        dataset = TrainingSet(X_train, y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, sampler=None, num_workers=0)
        
        if val > 0:
            val_dataset = TrainingSet(X_test, y_test)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, sampler=None, num_workers=0)
        else:
            val_loader = None

        self.model.train()
        if self.n_classes <= 2:
            criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        else:
            criterion = torch.nn.NLLLoss(reduction='mean')

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.9, patience=10, verbose=False, threshold=0.0001,
            threshold_mode='rel', cooldown=5, min_lr=1e-5, eps=1e-08)

        n_epochs_without_improvement = 0
        state_dict_history = []
        for e in range(epochs):

            # Training error
            total_error = 0
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                y_hat = self.model.forward(x)
                if self.n_classes > 2:
                    y_hat = torch.log_softmax(y_hat, dim=1)
                else:
                    y_hat = torch.squeeze(y_hat)

                loss = criterion(y_hat, y)
                # loss = criterion(y_hat, y.float())
                for loss_callback in self.loss_callbacks:
                    loss = loss + loss_callback()  # Add regularisation terms
                loss.backward()
                optimizer.step()
                total_error += loss.item()
            scheduler.step(total_error)

            if val_loader is not None:
                # Validation error
                with torch.no_grad():
                    val_total_error = 0
                    for x, y in val_loader:
                        x = x.to(device)
                        y = y.to(device)
                        y_hat = self.model.forward(x)
                        if self.n_classes > 2:
                            y_hat = torch.log_softmax(y_hat, dim=1)
                        else:
                            y_hat = torch.squeeze(y_hat)
                        loss = criterion(y_hat, y)
                        # loss = criterion(y_hat, y.float())
                        val_total_error += loss.item()

                # Keep track of parameters
                state_dict_history.append((val_total_error, self.model.state_dict()))
                if len(state_dict_history) >= 2:
                    if state_dict_history[-1][0] >= state_dict_history[-2][0]:
                        n_epochs_without_improvement += 1
                        if n_epochs_without_improvement >= 5:
                            break
                    else:
                        n_epochs_without_improvement = 0

        # Restore best parameters
        if val > 0:
            i = np.argmin([error for error, state_dict in state_dict_history])
            self.model.load_state_dict(state_dict_history[i][1])

        self.model.eval()
        self.trained = True
    
    def predict(self, X, device='cpu'):
        self.model.eval()
        dataset = TestSet(X)
        loader = DataLoader(dataset, batch_size=len(X), shuffle=False, sampler=None, num_workers=0)
        predictions = []
        for sample in loader:
            x = sample.to(device)
            y_pred = self.model.forward(x)
            if self.n_classes <= 2:
                y_pred = torch.sigmoid(y_pred)
            else:
                y_pred = torch.softmax(y_pred, dim=1)
            predictions += y_pred.data.squeeze().tolist()
        return np.array(predictions)

    def feature_importance(self, X):
        X = torch.FloatTensor(X)
        X.requires_grad_()
        ig = captum.attr.Saliency(self.model)
        attr = ig.attribute(X, target=0, abs=True)
        scores = attr.detach().numpy()
        return np.mean(np.abs(scores), axis=0)

    @staticmethod
    def create(dataset_name, n_input, n_classes, arch='nn'):
        # assert dataset_name in {'dag', 'xor', 'ring', 'ring+xor', 'ring+xor+sum'}
        loss_callbacks = []
        if arch == 'nn':
            model = Model(n_input, n_classes)
        elif arch == 'cancelout-sigmoid':
            model = ModelWithCancelOut(n_input, n_classes, cancel_out_activation='sigmoid')
            loss_callbacks.append(lambda: model.cancel_out.weight_loss())
        elif arch == 'cancelout-softmax':
            model = ModelWithCancelOut(n_input, n_classes, cancel_out_activation='softmax')
        elif arch == 'deeppink':
            _lambda = 0.05 * np.sqrt(2.0 * np.log(n_input) / 1000)
            model = DeepPINK(Model(n_input, n_classes), n_input)
            for layer in model.children():
                if isinstance(layer, torch.nn.Linear):
                    loss_callbacks.append(lambda: _lambda * torch.sum(torch.abs(layer.weight)))
        else:
            raise NotImplementedError(f'Unknown neural architecture "{arch}"')
        wrapper = NNwrapper(model, n_classes)
        for loss_callback in loss_callbacks:
            wrapper.add_loss_callback(loss_callback)
        return wrapper
