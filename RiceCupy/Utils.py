import math

from Backend import xp as np


NUMBA_AVAILABLE = False
USE_NUMBA_ACCELERATION = False


def he_init(shape):
    fan_in = math.prod(shape[1:])
    return (np.random.randn(*shape) * np.sqrt(2.0 / fan_in)).astype(np.float32)


def pad_image(x, padding):
    if padding == 0:
        return x

    if x.ndim == 3:
        return np.pad(
            x,
            ((0, 0), (padding, padding), (padding, padding)),
            mode="constant",
        )

    if x.ndim == 4:
        return np.pad(
            x,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode="constant",
        )

    raise ValueError(f"Unsupported shape: {x.shape}")


def im2col_batch(x, filter_h, filter_w, stride=1):
    b, c, h, w = x.shape
    out_h = (h - filter_h) // stride + 1
    out_w = (w - filter_w) // stride + 1

    patches = []
    for fy in range(filter_h):
        y_max = fy + stride * out_h
        for fx in range(filter_w):
            x_max = fx + stride * out_w
            patches.append(x[:, :, fy:y_max:stride, fx:x_max:stride])

    cols = np.stack(patches, axis=2)
    cols = cols.transpose(0, 3, 4, 1, 2)
    return cols.reshape(b * out_h * out_w, c * filter_h * filter_w).astype(
        np.float32,
        copy=False,
    )


def col2im_batch(cols, x_shape, filter_h, filter_w, stride=1, padding=0):
    b, c, h, w = x_shape
    h_padded = h + 2 * padding
    w_padded = w + 2 * padding

    out_h = (h_padded - filter_h) // stride + 1
    out_w = (w_padded - filter_w) // stride + 1

    cols = cols.reshape(b, out_h, out_w, c, filter_h, filter_w)
    cols = cols.transpose(0, 3, 4, 5, 1, 2)

    x_padded = np.zeros((b, c, h_padded, w_padded), dtype=np.float32)

    for fy in range(filter_h):
        y_max = fy + stride * out_h
        for fx in range(filter_w):
            x_max = fx + stride * out_w
            x_padded[:, :, fy:y_max:stride, fx:x_max:stride] += cols[:, :, fy, fx]

    if padding > 0:
        return x_padded[:, :, padding:-padding, padding:-padding]

    return x_padded
