import torch


class CancelOut(torch.nn.Module):

    def __init__(self, input_size, activation='sigmoid', lambda1=0.2, lambda2=0.1, beta=1):
        torch.nn.Module.__init__(self)
        assert activation in {'sigmoid', 'softmax'}
        self.activation = activation
        self.input_size = input_size
        self.beta = beta
        self.weights = torch.nn.Parameter(torch.zeros(input_size, requires_grad=True).float() + self.beta)
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def forward(self, x):
        w = self.get_weights()
        return w.unsqueeze(0) * x

    def weight_loss(self):
        w = self.weights / self.input_size
        loss = -self.lambda1 * torch.var(w)
        # loss = loss + self.lambda2 * torch.norm(w, 1)
        loss = loss + self.lambda2 * torch.sum(w)
        return loss

    def get_weights(self):
        if self.activation == 'sigmoid':
            w = torch.sigmoid(self.weights)
        else:
            w = torch.softmax(self.weights, 0)
        return w
