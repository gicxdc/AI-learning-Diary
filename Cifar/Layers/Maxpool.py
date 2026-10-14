import numpy as np

from Layers.Module import Module
from Utils import im2col_batch, col2im_batch


class MaxPool2D(Module):

    def __init__(self, size=2, stride=2):

        self.size = size
        self.stride = stride

        self.input_shape = None

        self.max_idx = None
        self.out_h = None
        self.out_w = None

    # =====================================
    # FORWARD
    # =====================================

    def forward(self, x):

        self.input_shape = x.shape

        B, C, H, W = x.shape

        # ---------------------------------
        # OUTPUT SIZE
        # ---------------------------------

        self.out_h = (
            (H - self.size)
            // self.stride
            + 1
        )

        self.out_w = (
            (W - self.size)
            // self.stride
            + 1
        )

        # ---------------------------------
        # IM2COL
        # ---------------------------------

        X_col = im2col_batch(
            x,
            self.size,
            self.size,
            self.stride
        )

        # ---------------------------------
        # RESHAPE
        # ---------------------------------

        X_col = X_col.reshape(
            B,
            self.out_h * self.out_w,
            C,
            self.size * self.size
        )

        X_col = X_col.transpose(
            0,
            2,
            3,
            1
        )

        # Shape:
        # (B, C, pool_size*pool_size, num_windows)

        # ---------------------------------
        # MAX + ARGMAX
        # ---------------------------------

        out = np.max(
            X_col,
            axis=2
        )

        self.max_idx = np.argmax(
            X_col,
            axis=2
        )

        # ---------------------------------
        # RESHAPE OUTPUT
        # ---------------------------------

        out = out.reshape(
            B,
            C,
            self.out_h,
            self.out_w
        )

        return out

    # =====================================
    # BACKWARD
    # =====================================

    def backward(self, dout):

        B, C, H, W = self.input_shape

        # ---------------------------------
        # RESHAPE DOUT
        # ---------------------------------

        dout_flat = dout.reshape(
            B,
            C,
            -1
        )

        # ---------------------------------
        # BUILD dX_col
        # ---------------------------------

        dX_col = np.zeros(
            (
                B,
                C,
                self.size * self.size,
                self.out_h * self.out_w
            ),
            dtype=dout.dtype
        )

        # ---------------------------------
        # VECTORIZED INDEXING
        # ---------------------------------

        b_idx = np.arange(B)[:, None, None]

        c_idx = np.arange(C)[None, :, None]

        w_idx = np.arange(
            self.out_h * self.out_w
        )[None, None, :]

        dX_col[
            b_idx,
            c_idx,
            self.max_idx,
            w_idx
        ] = dout_flat

        # ---------------------------------
        # RESHAPE FOR COL2IM
        # ---------------------------------

        dX_col = dX_col.transpose(
            0,
            3,
            1,
            2
        )

        dX_col = dX_col.reshape(
            B * self.out_h * self.out_w,
            C * self.size * self.size
        )

        # ---------------------------------
        # COL2IM
        # ---------------------------------

        dx = col2im_batch(
            dX_col,
            self.input_shape,
            self.size,
            self.size,
            self.stride
        )

        return dx