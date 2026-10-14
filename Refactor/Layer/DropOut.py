# layers/dropout.py

import numpy as np


class Dropout:

    def __init__(self, p=0.5):

        # keep probability
        self.p = p

    def forward(self, x, training=True):

        if training:

            # =====================================
            # inverted dropout
            # =====================================
            self.mask = (
                np.random.rand(*x.shape) < self.p
            ) / self.p

            return x * self.mask

        else:

            # =====================================
            # inference mode
            # =====================================
            return x

    def backward(self, dout):

        return dout * self.mask