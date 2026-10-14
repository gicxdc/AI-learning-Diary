from Layers.Conv import Conv2D
from Layers.Batchnorm import BatchNorm2D
from Layers.Relu import ReLU
from Layers.Maxpool import MaxPool2D
from Layers.Flatten import Flatten
from Layers.Linear import Linear
from Layers.Dropout import Dropout


class CNN:

    def __init__(self):

        # block 1
        self.conv1 = Conv2D(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.bn1 = BatchNorm2D(16)

        self.relu1 = ReLU()

        self.pool1 = MaxPool2D(2, 2)

        # block 2
        self.conv2 = Conv2D(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.bn2 = BatchNorm2D(32)

        self.relu2 = ReLU()

        self.pool2 = MaxPool2D(2, 2)

        # classifier
        self.flatten = Flatten()

        self.fc1 = Linear(
            32 * 7 * 7,
            128
        )

        self.relu3 = ReLU()

        self.dropout = Dropout(0.5)

        self.fc2 = Linear(
            128,
            10
        )

        # registry
        self.layers = [

            self.conv1,
            self.bn1,
            self.relu1,
            self.pool1,

            self.conv2,
            self.bn2,
            self.relu2,
            self.pool2,

            self.flatten,

            self.fc1,
            self.relu3,
            self.dropout,
            self.fc2
        ]

    def forward(
        self,
        x,
        training=True
    ):

        x = self.conv1.forward(x)

        x = self.bn1.forward(
            x,
            training
        )

        x = self.relu1.forward(x)

        x = self.pool1.forward(x)

        x = self.conv2.forward(x)

        x = self.bn2.forward(
            x,
            training
        )

        x = self.relu2.forward(x)

        x = self.pool2.forward(x)

        x = self.flatten.forward(x)

        x = self.fc1.forward(x)

        x = self.relu3.forward(x)

        x = self.dropout.forward(
            x,
            training
        )

        x = self.fc2.forward(x)

        return x

    def backward(self, dout):

        dout = self.fc2.backward(dout)

        dout = self.dropout.backward(dout)

        dout = self.relu3.backward(dout)

        dout = self.fc1.backward(dout)

        dout = self.flatten.backward(dout)

        dout = self.pool2.backward(dout)

        dout = self.relu2.backward(dout)

        dout = self.bn2.backward(dout)

        dout = self.conv2.backward(dout)

        dout = self.pool1.backward(dout)

        dout = self.relu1.backward(dout)

        dout = self.bn1.backward(dout)

        dout = self.conv1.backward(dout)

        return dout

    def parameters(self):

        params = []

        for layer in self.layers:

            params.extend(
                layer.parameters()
            )

        return params