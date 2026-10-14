import numpy as np

from Models.Model import Model
from Layers.Conv import Conv2D
from Layers.Batchnorm import BatchNorm2D
from Layers.Relu import ReLU
from Layers.Linear import Linear
from Layers.Residual import ResidualBlock


class ResNet9(Model):

    def __init__(self, num_classes=10, width_multiplier=1.0):

        super().__init__()

        c1 = max(16, int(64 * width_multiplier))
        c2 = max(32, int(128 * width_multiplier))
        c3 = max(64, int(256 * width_multiplier))

        self.stem_conv = Conv2D(3, c1, 3, padding=1)
        self.stem_bn = BatchNorm2D(c1)
        self.stem_relu = ReLU()

        self.block1 = ResidualBlock(c1, c1)
        self.block2 = ResidualBlock(c1, c2, stride=2)
        self.block3 = ResidualBlock(c2, c3, stride=2)
        self.block4 = ResidualBlock(c3, c3)

        self.fc = Linear(c3, num_classes)

        self.layers = [
            self.stem_conv,
            self.stem_bn,
            self.stem_relu,
            self.block1,
            self.block2,
            self.block3,
            self.block4,
            self.fc,
        ]

        self.gap_shape = None

    def forward(self, x):

        x = self.stem_conv.forward(x)
        x = self.stem_bn.forward(x, self.training)
        x = self.stem_relu.forward(x)

        x = self.block1.forward(x, self.training)
        x = self.block2.forward(x, self.training)
        x = self.block3.forward(x, self.training)
        x = self.block4.forward(x, self.training)

        self.gap_shape = x.shape
        x = np.mean(x, axis=(2, 3))

        x = self.fc.forward(x)

        return x

    def backward(self, dout):

        dout = self.fc.backward(dout)

        B, C, H, W = self.gap_shape
        dout = dout[:, :, None, None] / (H * W)
        dout = np.broadcast_to(dout, (B, C, H, W)).copy()

        dout = self.block4.backward(dout)
        dout = self.block3.backward(dout)
        dout = self.block2.backward(dout)
        dout = self.block1.backward(dout)

        dout = self.stem_relu.backward(dout)
        dout = self.stem_bn.backward(dout)
        dout = self.stem_conv.backward(dout)

        return dout

    def parameters(self):

        return super().parameters()
