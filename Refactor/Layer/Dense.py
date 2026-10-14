import numpy as np
from utils.init import he_init


class Dense:

    def __init__(self, in_features, out_features):

        self.W = he_init((out_features, in_features))
        self.b = np.zeros((out_features, 1))

    def forward(self, x):

        self.x = x
        return self.W @ x + self.b

    def backward(self, dout):

        self.dW = dout @ self.x.T
        self.db = np.sum(dout, axis=1, keepdims=True)

        dx = self.W.T @ dout

        return dx