import numpy as np
import torch


class LocallyConnected1d(torch.nn.Module):

    def __init__(self, n_filters, kernel_size, bias=True):
        torch.nn.Module.__init__(self)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        weight = torch.zeros((self.n_filters, self.kernel_size), requires_grad=True).float()
        self.weight = torch.nn.Parameter(weight)
        if bias:
            bias = torch.zeros(self.n_filters, requires_grad=True).float()
            self.bias = torch.nn.Parameter(bias)
        else:
            self.bias = None
        self.init_weights()

    def forward(self, X):
        assert len(X.size()) == 3
        w = self.weight.unsqueeze(0)
        X = torch.sum(X * w, axis=2)
        if self.bias is not None:
            X = X + self.bias.unsqueeze(0)
        return X.unsqueeze(2)

    def init_weights(self):
        self.weight.data.fill_(0.1)
        if self.bias is not None:
            self.bias.data.fill_(0)


class DeepPINK(torch.nn.Module):

    def __init__(self, model, n_input):
        torch.nn.Module.__init__(self)
        self.n_input = n_input
        self.lc1 = LocallyConnected1d(n_input, 2)
        self.lc2 = LocallyConnected1d(n_input, 1)
        torch.nn.init.xavier_normal_(self.lc2.weight)
        self.model = model

    def forward(self, X):
        X = self.lc1(X)
        X = self.lc2(X)
        X = torch.squeeze(X, 2)
        X = self.model(X)
        return X

    def get_weights(self):
        W0 = self.lc2.weight.data.numpy()
        W_acc = None
        W = []
        for layer in self.model.layers:
            if isinstance(layer, torch.nn.Linear):
                W.append(layer.weight.data.numpy().T)
                W_acc = W[-1] if (W_acc is None) else np.dot(W_acc, W[-1])
        w = np.squeeze(W0 * W_acc)
        z = self.lc1.weight.data.numpy()[:, 0] * w
        z_tilde = self.lc1.weight.data.numpy()[:, 1] * w
        return z ** 2. - z_tilde ** 2.
