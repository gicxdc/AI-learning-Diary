import numpy as np
from la import Module


class BatchNorm2D(Module):

    def __init__(self, channels, eps=1e-5, momentum=0.1):

        self.eps = eps
        self.momentum = momentum

        # trainable parameters
        self.gamma = np.ones((1, channels, 1, 1), dtype=np.float32)
        self.beta = np.zeros((1, channels, 1, 1), dtype=np.float32)

        # running stats
        self.running_mean = np.zeros((1, channels, 1, 1), dtype=np.float32)
        self.running_var = np.ones((1, channels, 1, 1), dtype=np.float32)

        # cache
        self.cache = None

        # grads
        self.dgamma = None
        self.dbeta = None

    # =====================================
    # FORWARD
    # =====================================

    def forward(self, x, training=True):

        if training:

            mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            var = np.var(x, axis=(0, 2, 3), keepdims=True)

            # FIXED running stats update
            self.running_mean = (
                (1 - self.momentum) * self.running_mean
                + self.momentum * mean
            )

            self.running_var = (
                (1 - self.momentum) * self.running_var
                + self.momentum * var
            )

        else:
            mean = self.running_mean
            var = self.running_var

        x_mu = x - mean
        std_inv = 1.0 / np.sqrt(var + self.eps)

        x_hat = x_mu * std_inv

        out = self.gamma * x_hat + self.beta

        # cache for backward
        self.cache = (x, x_hat, mean, var, x_mu, std_inv)

        return out

    # =====================================
    # BACKWARD (FULL CORRECT VERSION)
    # =====================================

    def backward(self, dout):

        x, x_hat, mean, var, x_mu, std_inv = self.cache

        B, C, H, W = x.shape
        N = B * H * W  # elements per channel

        # ---------------------
        # gamma / beta grads
        # ---------------------

        self.dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)
        self.dgamma = np.sum(dout * x_hat, axis=(0, 2, 3), keepdims=True)

        # ---------------------
        # upstream gradient
        # ---------------------

        dx_hat = dout * self.gamma

        # ---------------------
        # intermediate grads
        # ---------------------

        dvar = np.sum(
            dx_hat * x_mu * -0.5 * (var + self.eps) ** (-1.5),
            axis=(0, 2, 3),
            keepdims=True
        )

        dmean = np.sum(
            dx_hat * (-std_inv),
            axis=(0, 2, 3),
            keepdims=True
        )

        dmean += dvar * np.sum(
            -2.0 * x_mu,
            axis=(0, 2, 3),
            keepdims=True
        ) / N

        # ---------------------
        # final gradient
        # ---------------------

        dx = (
            dx_hat * std_inv
            + (dvar * 2.0 * x_mu / N)
            + (dmean / N)
        )

        return dx

    # =====================================
    # PARAMETERS
    # =====================================

    def parameters(self):
        return [
            (self.gamma, self.dgamma),
            (self.beta, self.dbeta)
        ]