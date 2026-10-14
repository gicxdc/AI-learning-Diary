import numpy as np
from utils.init import he_init
from utils.im2col import im2col, col2im
from utils.im2col import pad_image


class Conv2D:

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1):

        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        self.W = he_init(
            (out_channels, in_channels,
             kernel_size, kernel_size)
        )

        self.b = np.zeros((out_channels, 1))

    def forward(self, x):

        self.x = x

        B, C, H, W = x.shape

        outputs = []
        self.caches = []

        for b in range(B):

            image = x[b]

            image_padded = pad_image(image, self.padding)

            X_col = im2col(
                image_padded,
                self.kernel_size,
                self.kernel_size,
                self.stride
            )

            W_col = self.W.reshape(self.W.shape[0], -1)

            out = W_col @ X_col + self.b

            out_h = (
                (image_padded.shape[1] - self.kernel_size)
                // self.stride + 1
            )

            out_w = (
                (image_padded.shape[2] - self.kernel_size)
                // self.stride + 1
            )

            out = out.reshape(self.W.shape[0], out_h, out_w)

            outputs.append(out)

            self.caches.append(
                (image, image_padded, X_col)
            )

        return np.stack(outputs)

    def backward(self, dout):

        B = dout.shape[0]

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        dx = np.zeros_like(self.x)

        for b in range(B):

            image, image_padded, X_col = self.caches[b]

            dout_col = dout[b].reshape(self.W.shape[0], -1)

            W_col = self.W.reshape(self.W.shape[0], -1)

            dW = dout_col @ X_col.T
            db = np.sum(dout_col, axis=1, keepdims=True)

            dX_col = W_col.T @ dout_col

            dX_padded = col2im(
                dX_col,
                image_padded.shape,
                self.kernel_size,
                self.kernel_size,
                self.stride
            )

            if self.padding > 0:
                dx[b] = dX_padded[
                    :,
                    self.padding:-self.padding,
                    self.padding:-self.padding
                ]
            else:
                dx[b] = dX_padded

            self.dW += dW.reshape(self.W.shape)
            self.db += db

        self.dW /= B
        self.db /= B

        return dx