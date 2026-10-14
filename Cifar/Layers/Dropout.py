import numpy as np
from Layers.Module import Module


class Dropout(Module):

    def __init__(self, p=0.5):

        # p = dropout probability
        self.p = p

        self.mask = None

    # =====================================
    # FORWARD
    # =====================================

    def forward(self, x, training=True):

        if not training:
            return x

        keep_prob = 1 - self.p
        self.mask = (np.random.rand(*x.shape) < keep_prob) / keep_prob

        return x * self.mask

    # =====================================
    # BACKWARD
    # =====================================

    def backward(self, dout):

        return dout * self.mask