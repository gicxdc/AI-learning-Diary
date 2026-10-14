from Layers.Module import Module

class Flatten(Module):

    def __init__(self):
        self.original_shape = None

    def forward(self, x):
        self.original_shape = x.shape
        B = x.shape[0]
        return x.reshape(B, -1)   

    def backward(self, dout):
        return dout.reshape(self.original_shape)