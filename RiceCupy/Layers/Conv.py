from Backend import xp as np

from Layers.Module import Module
from Utils import he_init, im2col_batch, col2im_batch, pad_image


class Conv2D(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):

        self.stride = stride
        self.padding = padding

        # =====================================
        # PARAMETERS
        # =====================================

        self.weight = he_init((out_channels, in_channels, kernel_size, kernel_size))

        self.bias = np.zeros(out_channels, dtype=np.float32)

        # =====================================
        # CACHE
        # =====================================

        self.input_shape = None
        self.X_col = None

        # =====================================
        # GRADS
        # =====================================

        self.dW = np.zeros_like(self.weight)
        self.db = np.zeros_like(self.bias)

    # =====================================
    # FORWARD
    # =====================================

    def forward(self, x):

        self.input_shape = x.shape

        B, C, H, W = x.shape

        F, _, FH, FW = self.weight.shape

        # ---------------------------------
        # PAD INPUT
        # ---------------------------------

        x_padded = pad_image(x, self.padding)

        # ---------------------------------
        # IM2COL
        # ---------------------------------

        self.X_col = im2col_batch(x_padded, FH, FW, self.stride)

        # ---------------------------------
        # RESHAPE WEIGHTS
        # ---------------------------------

        W_col = self.weight.reshape(F, -1)

        # ---------------------------------
        # GEMM
        # ---------------------------------

        out = self.X_col @ W_col.T

        out += self.bias

        # ---------------------------------
        # OUTPUT SIZE
        # ---------------------------------

        out_h = (H + 2 * self.padding - FH) // self.stride + 1

        out_w = (W + 2 * self.padding - FW) // self.stride + 1

        # ---------------------------------
        # RESHAPE OUTPUT
        # ---------------------------------

        out = out.reshape(B, out_h, out_w, F)

        out = out.transpose(0, 3, 1, 2)

        return out

    # =====================================
    # BACKWARD
    # =====================================

    def backward(self, dout):

        B = dout.shape[0]

        F, C, FH, FW = self.weight.shape

        # ---------------------------------
        # RESHAPE DOUT
        # ---------------------------------

        dout_col = dout.transpose(0, 2, 3, 1)

        dout_col = dout_col.reshape(-1, F)

        # ---------------------------------
        # RESHAPE WEIGHTS
        # ---------------------------------

        W_col = self.weight.reshape(F, -1)

        # =================================
        # dW
        # =================================

        self.dW[:] = (dout_col.T @ self.X_col).reshape(self.weight.shape)

        # =================================
        # db
        # =================================

        self.db[:] = np.sum(dout_col, axis=0)

        # =================================
        # dX_col
        # =================================

        dX_col = dout_col @ W_col

        # =================================
        # COL2IM
        # =================================

        dx = col2im_batch(dX_col, self.input_shape, FH, FW, self.stride, self.padding)

        return dx

    # =====================================
    # PARAMETERS
    # =====================================

    def parameters(self):

        return [(self.weight, self.dW), (self.bias, self.db)]

