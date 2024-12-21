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

import tqdm
import captum.attr
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.cancelout import CancelOut
from src.deeppink import DeepPINK
from src.utils import TrainingSet, TestSet
from src.sam import SharpnessAwareMinimizer


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


"""
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
"""
class Model(torch.nn.Module):

    def __init__(self, input_size, n_classes, latent_size=58, gaussian_noise=0.7466805127272365, dropout=0.04308691548552568, n_hidden_layers=5, layer_norm=0, activation='mish'):
        torch.nn.Module.__init__(self)
        n_out = 1 if (n_classes <= 2) else n_classes
        layers = []
        if gaussian_noise > 0:
            layers.append(GaussianNoise(gaussian_noise))
        inplace = False
        for k in range(n_hidden_layers):

            if dropout > 0:
                layers.append(torch.nn.Dropout(p=dropout, inplace=inplace))

            if k == 0:
                layers.append(torch.nn.Linear(input_size, latent_size))
            else:
                layers.append(torch.nn.Linear(latent_size, latent_size))

            if layer_norm:
                layers.append(torch.nn.LayerNorm(latent_size))

            if activation == 'relu':
                layers.append(torch.nn.ReLU(inplace=inplace))
            elif activation == 'leakyrelu':
                layers.append(torch.nn.LeakyReLU(0.2, inplace=inplace))
            elif activation == 'prelu':
                layers.append(torch.nn.PReLU(latent_size))
            elif activation == 'tanh':
                layers.append(torch.nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(torch.nn.Sigmoid())
            elif activation == 'mish':
                layers.append(torch.nn.Mish(inplace=inplace))
            elif activation == 'selu':
                layers.append(torch.nn.SELU(inplace=inplace))
            else:
                layers.append(torch.nn.Hardswish(inplace=inplace))

        layers.append(torch.nn.Linear(latent_size, n_out))

        self.layers = torch.nn.Sequential(*layers)
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

    def fit(
            self,
            X,
            Y,
            device='cpu',
            learning_rate=0.0017601777068292975,  # 0.0005
            epochs=416,  # 1000
            batch_size=56,  # 64
            weight_decay=0.00048519293899787247,  # 1e-5
            val=0.2,
            early_stopping_patience=66,  # 5
            optimizer='adagrad',  # 'adam'
            sam_type='no-sam'  # 'no-sam'
    ):

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

        if optimizer == 'adam':
            optimizer_class = torch.optim.Adam
        elif optimizer == 'sgd':
            optimizer_class = torch.optim.SGD
        elif optimizer == 'rmsprop':
            optimizer_class = torch.optim.RMSprop
        elif optimizer == 'adamw':
            optimizer_class = torch.optim.AdamW
        else:
            optimizer_class = torch.optim.Adagrad
        if sam_type == 'no-sam':
            optimizer = optimizer_class(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif sam_type == 'sam':
            optimizer = SharpnessAwareMinimizer(self.model.parameters(), optimizer_class, lr=learning_rate, weight_decay=weight_decay, adaptive=False)
        else:
            optimizer = SharpnessAwareMinimizer(self.model.parameters(), optimizer_class, lr=learning_rate, weight_decay=weight_decay, adaptive=True)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.9, patience=10, verbose=False, threshold=0.0001,
            threshold_mode='rel', cooldown=5, min_lr=1e-5, eps=1e-08)

        n_epochs_without_improvement = 0
        state_dict_history = []
        pbar = tqdm.tqdm(range(epochs))
        for e in pbar:

            # Training error
            total_error = 0
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()

                def closure():
                    y_hat = self.model.forward(x)
                    if self.n_classes > 2:
                        y_hat = torch.log_softmax(y_hat, dim=1)
                    else:
                        y_hat = y_hat.reshape(len(y_hat))

                    try:
                        loss = criterion(y_hat, y)
                    except RuntimeError:
                        loss = criterion(y_hat, y.float())

                    for loss_callback in self.loss_callbacks:
                        loss = loss + loss_callback()  # Add regularisation terms
                    loss.backward()
                    return loss

                loss = closure()
                if sam_type == 'no-sam':    
                    optimizer.step()
                else:
                    optimizer.step(closure=closure)

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
                            y_hat = y_hat.reshape(len(y_hat))
                        try:
                            loss = criterion(y_hat, y)
                        except RuntimeError:
                            loss = criterion(y_hat, y.float())
                        val_total_error += loss.item()

                # Keep track of parameters
                state_dict_history.append((val_total_error, self.model.state_dict()))
                if len(state_dict_history) >= 2:
                    if state_dict_history[-1][0] >= state_dict_history[-2][0]:
                        n_epochs_without_improvement += 1
                        if n_epochs_without_improvement >= early_stopping_patience:  # 5
                            break
                    else:
                        n_epochs_without_improvement = 0

                pbar.set_description(str(total_error))

        # Restore best parameters
        if val > 0:
            i = np.argmin([error for error, state_dict in state_dict_history])
            self.model.load_state_dict(state_dict_history[i][1])

        self.model.eval()
        self.trained = True
    
    def predict_proba(self, X, device='cpu'):
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

    def predict(self, X, device='cpu'):
        y_proba = self.predict_proba(X, device=device)
        if len(y_proba.shape) == 2:
            return np.argmax(y_proba, axis=1)
        else:
            return (y_proba > 0.5).astype(int)

    def feature_importance(self, X):
        X = torch.FloatTensor(X)
        X.requires_grad_()
        ig = captum.attr.Saliency(self.model)
        attr = ig.attribute(X, target=0, abs=True)
        scores = attr.detach().numpy()
        return np.mean(np.abs(scores), axis=0)

    @staticmethod
    def create(dataset_name, n_input, n_classes, arch='nn'):
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
                # if isinstance(layer, torch.nn.Linear) and (layer.out_features > 1):
                if isinstance(layer, torch.nn.Linear):
                    loss_callbacks.append(lambda: _lambda * torch.sum(torch.abs(layer.weight)))
        else:
            raise NotImplementedError(f'Unknown neural architecture "{arch}"')
        wrapper = NNwrapper(model, n_classes)
        for loss_callback in loss_callbacks:
            wrapper.add_loss_callback(loss_callback)
        return wrapper
