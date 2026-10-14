from Layers.Module import Module
from Layers.Conv import Conv2D
from Layers.Batchnorm import BatchNorm2D
from Layers.Relu import ReLU


class ResidualBlock(Module):

    def __init__(self, in_channels, out_channels=None, stride=1):

        if out_channels is None:
            out_channels = in_channels

        self.conv1 = Conv2D(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = BatchNorm2D(out_channels)
        self.relu1 = ReLU()

        self.conv2 = Conv2D(out_channels, out_channels, 3, padding=1)
        self.bn2 = BatchNorm2D(out_channels)
        self.relu2 = ReLU()

        self.proj = None
        self.proj_bn = None

        if stride != 1 or in_channels != out_channels:
            self.proj = Conv2D(in_channels, out_channels, 1, stride=stride)
            self.proj_bn = BatchNorm2D(out_channels)

        self.layers = [
            self.conv1,
            self.bn1,
            self.conv2,
            self.bn2,
        ]

        if self.proj is not None:
            self.layers.extend([self.proj, self.proj_bn])

    def forward(self, x, training=True):

        shortcut = x

        if self.proj is not None:
            shortcut = self.proj.forward(shortcut)
            shortcut = self.proj_bn.forward(shortcut, training)

        out = self.conv1.forward(x)
        out = self.bn1.forward(out, training)
        out = self.relu1.forward(out)

        out = self.conv2.forward(out)
        out = self.bn2.forward(out, training)

        out = shortcut + out
        out = self.relu2.forward(out)

        return out

    def backward(self, dout):

        dout = self.relu2.backward(dout)

        shortcut_grad = dout

        if self.proj is not None:
            shortcut_grad = self.proj_bn.backward(shortcut_grad)
            shortcut_grad = self.proj.backward(shortcut_grad)

        out_grad = self.bn2.backward(dout)
        out_grad = self.conv2.backward(out_grad)

        out_grad = self.relu1.backward(out_grad)
        out_grad = self.bn1.backward(out_grad)
        out_grad = self.conv1.backward(out_grad)

        return shortcut_grad + out_grad

    def parameters(self):

        params = []

        for layer in self.layers:
            params.extend(layer.parameters())

        return params

