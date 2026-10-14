import numpy as np

from layers.conv import Conv2D
from layers.relu import ReLU
from layers.pool import MaxPool2D
from layers.batchnorm import BatchNorm2D
from layers.dense import Dense


class SimpleCNN:

    def __init__(self):

        self.conv1 = Conv2D(1, 16)
        self.bn1 = BatchNorm2D(16)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D()

        self.conv2 = Conv2D(16, 32)
        self.bn2 = BatchNorm2D(32)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D()

        self.fc = Dense(32 * 7 * 7, 10)

    def forward(self, x):

        x = self.conv1.forward(x)
        x = self.bn1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)

        x = self.conv2.forward(x)
        x = self.bn2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)

        self.shape_before_flatten = x.shape

        B = x.shape[0]

        x = x.reshape(B, -1).T

        x = self.fc.forward(x)

        return x

    def backward(self, dout):

        dout = self.fc.backward(dout)

        B = self.shape_before_flatten[0]

        dout = dout.T.reshape(self.shape_before_flatten)

        dout = self.pool2.backward(dout)
        dout = self.relu2.backward(dout)
        dout = self.bn2.backward(dout)
        dout = self.conv2.backward(dout)

        dout = self.pool1.backward(dout)
        dout = self.relu1.backward(dout)
        dout = self.bn1.backward(dout)
        dout = self.conv1.backward(dout)

    def parameters(self):

        return [
            self.conv1,
            self.conv2,
            self.bn1,
            self.bn2,
            self.fc
        ]