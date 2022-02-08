# -*- coding: utf-8 -*-
#
#  pred1.py
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

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class Model(torch.nn.Module):

    def __init__(self, input_size, latent_size=32, name='NN'):
        torch.nn.Module.__init__(self)
        self.name = name
        self.input_size = input_size
        self.latent_size = latent_size
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.latent_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.latent_size, 1))
            
    def forward(self, x):
        return self.layers(x)

    @staticmethod
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            print(f'Initializing weights... {m.__class__.__name__}')
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.1)


class MyDataset(Dataset):
    
    def __init__(self, X, Y):    
    
        assert len(Y) == len(X)        
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class MypredDataset(Dataset):
    
    def __init__(self, X):    
        self.X = torch.tensor(X, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


class NNwrapper:

    def __init__(self, model):
        if type(model) == str:
            self.model = torch.load(model)
        else:
            self.model = model

    def fit(self, X, Y, device, epochs=50, batch_size=20,
            save_model_every=300, weight_decay=1e-5, learning_rate=1e-3):
        t1 = time.time()
        dataset = MyDataset(X, Y)
        t2 = time.time()
        print("Dataset created in %.2fs" % (t2 - t1))
        self.model.train()
        criterion = torch.nn.BCEWithLogitsLoss()
        p = []
        for i in list(self.model.parameters()):
            p += list(i.data.cpu().numpy().flat)
        print(f'Number of parameters: {len(p)}')
        del p

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.3, patience=7, verbose=True, threshold=0.0001,
            threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

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
                loss.backward()
                optimizer.step()
                total_error += loss.item()
                i += batch_size
            end = time.time()
            if e % 10 == 0:
                print(f'epoch {e}, ERRORTOT: {total_error} ({end - start})')
            scheduler.step(total_error)
            if e % save_model_every == 0:
                print('Store model ', e)
            e += 1
    
    def predict(self, X, device):
        self.model.eval()
        print('Predicting...')
        dataset = MypredDataset(X)
        loader = DataLoader(dataset, batch_size=len(X), shuffle=False, sampler=None, num_workers=0)
        predictions = []
        for sample in loader:
            x = sample.to(device)
            y_pred = self.model.forward(x)
            predictions += y_pred.data.squeeze().tolist()
        return np.array(predictions)