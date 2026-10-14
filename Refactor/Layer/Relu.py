import numpy as np

class ReLU:

    def forward(self, x):

        self.x = x
        return np.maximum(0, x)

    def backward(self, dout):

        dx = dout.copy()
        dx[self.x <= 0] = 0
        return dx