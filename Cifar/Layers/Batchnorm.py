import numpy as np
from Layers.Module import Module


class BatchNorm2D(Module):

    def __init__(self, channels, eps=1e-5, momentum=0.1):

        self.eps = eps
        self.momentum = momentum

        self.gamma = np.ones((1, channels, 1, 1), dtype=np.float32)
        self.beta = np.zeros((1, channels, 1, 1), dtype=np.float32)

        self.running_mean = np.zeros((1, channels, 1, 1), dtype=np.float32)
        self.running_var = np.ones((1, channels, 1, 1), dtype=np.float32)

        self.x_hat = None
        self.var = None
        self.mean = None
        self.x = None

        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)

    def reset_running_stats(self):

        self.running_mean[:] = 0
        self.running_var[:] = 1

    def forward(self, x, training=True):

        self.x = x

        if training:

            mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            var = np.var(x, axis=(0, 2, 3), keepdims=True)

            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * mean
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * var

        else:
            mean = self.running_mean
            var = self.running_var

        self.mean = mean
        self.var = var

        self.x_hat = (x - mean) / np.sqrt(var + self.eps)

        out = self.gamma * self.x_hat + self.beta

        return out

    def backward(self, dout):

        B, C, H, W = self.x.shape
        N = B * H * W

        self.dgamma[:] = np.sum(dout * self.x_hat, axis=(0, 2, 3), keepdims=True)
        self.dbeta[:] = np.sum(dout, axis=(0, 2, 3), keepdims=True)

        dxhat = dout * self.gamma

        var_eps = self.var + self.eps
        std_inv = 1.0 / np.sqrt(var_eps)

        dx = (
            (1.0 / N)
            * std_inv
            * (
                N * dxhat
                - np.sum(dxhat, axis=(0, 2, 3), keepdims=True)
                - self.x_hat * np.sum(dxhat * self.x_hat, axis=(0, 2, 3), keepdims=True)
            )
        )

        return dx

    def parameters(self):
        return [(self.gamma, self.dgamma), (self.beta, self.dbeta)]
