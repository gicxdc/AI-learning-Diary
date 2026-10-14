# Models/AlexNet.py

from Models.Model import Model

from Layers.Conv import Conv2D
from Layers.Relu import ReLU
from Layers.Maxpool import MaxPool2D
from Layers.Flatten import Flatten
from Layers.Linear import Linear


class AlexNet(Model):

    def __init__(self):

        super().__init__()

        # =====================================
        # BLOCK 1
        # =====================================

        self.conv1 = Conv2D(3, 16, 3, padding=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(2, 2)

        # =====================================
        # BLOCK 2
        # =====================================

        self.conv2 = Conv2D(16, 32, 3, padding=1)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(2, 2)

        # =====================================
        # BLOCK 3
        # =====================================

        self.conv3 = Conv2D(32, 64, 3, padding=1)
        self.relu3 = ReLU()

        # =====================================
        # FLATTEN
        # =====================================

        self.flatten = Flatten()

        # =====================================
        # FULLY CONNECTED
        # =====================================

        self.fc1 = Linear(64 * 8 * 8, 128)
        self.relu4 = ReLU()

        self.fc2 = Linear(128, 10)

        # =====================================
        # LAYER REGISTRY
        # =====================================

        self.layers = [
            self.conv1, self.relu1, self.pool1,
            self.conv2, self.relu2, self.pool2,
            self.conv3, self.relu3,
            self.flatten,
            self.fc1, self.relu4,
            self.fc2
        ]

    # =====================================
    # FORWARD
    # =====================================

    def forward(self, x, training=None):

        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)

        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)

        x = self.conv3.forward(x)
        x = self.relu3.forward(x)

        x = self.flatten.forward(x)

        x = self.fc1.forward(x)
        x = self.relu4.forward(x)

        x = self.fc2.forward(x)

        return x

    # =====================================
    # BACKWARD
    # =====================================

    def backward(self, dout):

        dout = self.fc2.backward(dout)

        dout = self.relu4.backward(dout)
        dout = self.fc1.backward(dout)

        dout = self.flatten.backward(dout)

        dout = self.relu3.backward(dout)
        dout = self.conv3.backward(dout)

        dout = self.pool2.backward(dout)

        dout = self.relu2.backward(dout)
        dout = self.conv2.backward(dout)

        dout = self.pool1.backward(dout)

        dout = self.relu1.backward(dout)
        dout = self.conv1.backward(dout)

        return dout
