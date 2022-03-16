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

import time

import captum.attr
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.cancelout import CancelOut
from src.deeppink import DeepPINK
from src.utils import TrainingSet, TestSet


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
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

    def __init__(self, input_size, latent_size=64):
        torch.nn.Module.__init__(self)
        self.layers = torch.nn.Sequential(
            GaussianNoise(0.05),
            torch.nn.Linear(input_size, latent_size),
            torch.nn.Dropout(p=0.2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(latent_size, latent_size),
            torch.nn.Dropout(p=0.2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(latent_size, 1))
        self.apply(init_weights)
            
    def forward(self, x):
        return self.layers(x)


class ModelWithCancelOut(torch.nn.Module):

    def __init__(self, input_size, latent_size=32, cancel_out_activation='sigmoid'):
        torch.nn.Module.__init__(self)
        self.cancel_out = CancelOut(input_size, activation=cancel_out_activation)
        self.model = Model(input_size, latent_size=latent_size)

    def forward(self, x):
        x = self.cancel_out(x)
        return self.model(x)


class NNwrapper:

    def __init__(self, model):
        self.model = model
        self.loss_callbacks = []

    def add_loss_callback(self, func):
        self.loss_callbacks.append(func)

    def fit(self, X, Y, device='cpu', learning_rate=0.005, epochs=300, batch_size=64, weight_decay=1e-6):
        dataset = TrainingSet(X, Y)
        self.model.train()
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.9, patience=10, verbose=False, threshold=0.0001,
            threshold_mode='rel', cooldown=5, min_lr=1e-5, eps=1e-08)

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, sampler=None, num_workers=0)
        e = 1
        while e < epochs:
            total_error = 0
            i = 1
            start = time.time()
            for x, y in loader:
                x = x.to(device)
                y = y.to(device).unsqueeze(1)
                optimizer.zero_grad()
                y_hat = self.model.forward(x)
                loss = criterion(y_hat, y)
                for loss_callback in self.loss_callbacks:
                    loss = loss + loss_callback()  # Add regularisation terms
                loss.backward()
                optimizer.step()
                total_error += loss.item()
                i += batch_size
            end = time.time()
            #if e % 10 == 0:
            #    print(f'epoch {e}, ERRORTOT: {total_error} ({end - start})')
            scheduler.step(total_error)
            e += 1
        self.model.eval()
    
    def predict(self, X, device='cpu'):
        self.model.eval()
        print('Predicting...')
        dataset = TestSet(X)
        loader = DataLoader(dataset, batch_size=len(X), shuffle=False, sampler=None, num_workers=0)
        predictions = []
        for sample in loader:
            x = sample.to(device)
            y_pred = self.model.forward(x)
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
    def create(dataset_name, n_input, arch='nn'):
        assert dataset_name in {'xor', 'ring', 'ring+xor', 'ring+xor+sum'}
        assert arch in {'nn', 'cancelout-sigmoid', 'cancelout-softmax', 'deeppink-2o'}
        loss_callbacks = []
        if arch == 'nn':
            model = Model(n_input)
        elif arch == 'cancelout-sigmoid':
            model = ModelWithCancelOut(n_input, cancel_out_activation='sigmoid')
            loss_callbacks.append(lambda: model.cancel_out.weight_loss())
        elif arch == 'cancelout-softmax':
            model = ModelWithCancelOut(n_input, cancel_out_activation='softmax')
        elif arch == 'deeppink-2o':
            _lambda = 0.05 * np.sqrt(2.0 * np.log(n_input) / 1000)
            model = DeepPINK(Model(n_input), n_input)
            for layer in model.children():
                if isinstance(layer, torch.nn.Linear):
                    loss_callbacks.append(lambda: _lambda * torch.sum(torch.abs(layer.weight)))
        else:
            raise NotImplementedError(f'Unknown neural architecture "{arch}"')
        wrapper = NNwrapper(model)
        for loss_callback in loss_callbacks:
            wrapper.add_loss_callback(loss_callback)
        return wrapper
