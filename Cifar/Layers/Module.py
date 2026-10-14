class Module:

    def __init__(self):
        pass

    def forward(self, x):
        raise NotImplementedError

    def backward(self, dout):
        raise NotImplementedError

    def parameters(self):
        return []