import numpy as np
from utils.im2col import im2col
from utils.im2col import col2im


class MaxPool2D:

    def __init__(self,
                 kernel_size=2,
                 stride=2):

        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):

        self.x_shape = x.shape

        B, C, H, W = x.shape

        outputs = []

        self.caches = []

        for b in range(B):

            image = x[b]

            # =====================================
            # im2col
            # =====================================
            X_col = im2col(
                image,
                self.kernel_size,
                self.kernel_size,
                self.stride
            )

            # =====================================
            # reshape:
            # (C*k*k, N) -> (C, k*k, N)
            # =====================================
            X_col = X_col.reshape(
                C,
                self.kernel_size * self.kernel_size,
                -1
            )

            # =====================================
            # max
            # =====================================
            out = np.max(X_col, axis=1)

            # save argmax for backward
            max_idx = np.argmax(X_col, axis=1)

            out_h = (
                (H - self.kernel_size)
                // self.stride + 1
            )

            out_w = (
                (W - self.kernel_size)
                // self.stride + 1
            )

            out = out.reshape(C, out_h, out_w)

            outputs.append(out)

            self.caches.append(
                (X_col, max_idx)
            )

        return np.stack(outputs)

    def backward(self, dout):

        B, C, out_h, out_w = dout.shape

        dx = np.zeros(self.x_shape)

        for b in range(B):

            X_col, max_idx = self.caches[b]

            dout_flat = dout[b].reshape(C, -1)

            # =====================================
            # gradient col buffer
            # =====================================
            dX_col = np.zeros_like(X_col)

            c_idx = np.arange(C)[:, None]
            w_idx = np.arange(dout_flat.shape[1])[None, :]

            # scatter gradients
            dX_col[c_idx, max_idx, w_idx] = dout_flat

            # =====================================
            # reshape for col2im
            # =====================================
            dX_col = dX_col.reshape(
                C * self.kernel_size * self.kernel_size,
                -1
            )

            dx[b] = col2im(
                dX_col,
                self.x_shape[1:],
                self.kernel_size,
                self.kernel_size,
                self.stride
            )

        return dx