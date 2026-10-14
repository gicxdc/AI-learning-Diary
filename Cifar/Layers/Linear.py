import numpy as np
from Layers.Module import Module
from Utils import he_init


class Linear(Module):

    def __init__(self, in_features, out_features):

        self.weight = he_init((out_features, in_features))
        self.bias = np.zeros(out_features, dtype=np.float32)

        self.x = None
        self.dW = np.zeros_like(self.weight)
        self.db = np.zeros_like(self.bias)

    def forward(self, x):
        self.x = x
        return x @ self.weight.T + self.bias

    def backward(self, dout):

        self.dW[:] = dout.T @ self.x
        self.db[:] = np.sum(dout, axis=0)

        dx = dout @ self.weight
        return dx

    def parameters(self):
        return [(self.weight, self.dW), (self.bias, self.db)]
