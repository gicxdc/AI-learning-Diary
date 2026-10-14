class Optimizer:

    def __init__(self, layers, lr):

        self.layers = layers
        self.lr = lr

    def step(self):
        raise NotImplementedError