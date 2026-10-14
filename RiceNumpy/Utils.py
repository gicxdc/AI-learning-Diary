import numpy as np
from numpy.lib.stride_tricks import as_strided

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def wrapper(func):
            return func

        return wrapper

    def prange(*args):
        return range(*args)

USE_NUMBA_ACCELERATION = NUMBA_AVAILABLE


def he_init(shape):
    fan_in = np.prod(shape[1:])
    return (np.random.randn(*shape) * np.sqrt(2.0 / fan_in)).astype(np.float32)


def pad_image(x, padding):

    if padding == 0:
        return x

    # ---------------------------------
    # SINGLE IMAGE
    # (C,H,W)
    # ---------------------------------

    if x.ndim == 3:

        return np.pad(
            x, ((0, 0), (padding, padding), (padding, padding)), mode="constant"
        )

    # ---------------------------------
    # BATCH
    # (B,C,H,W)
    # ---------------------------------

    elif x.ndim == 4:

        return np.pad(
            x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant"
        )

    else:

        raise ValueError(f"Unsupported shape: {x.shape}")


def im2col(image, filter_h, filter_w, stride=1):

    C, H, W = image.shape

    out_h = (H - filter_h) // stride + 1
    out_w = (W - filter_w) // stride + 1

    cols = []

    for y in range(out_h):
        for x in range(out_w):

            patch = image[
                :,
                y * stride : y * stride + filter_h,
                x * stride : x * stride + filter_w,
            ]

            cols.append(patch.reshape(-1))

    return np.stack(cols, axis=1)


def col2im(cols, image_shape, filter_h, filter_w, stride=1):

    C, H, W = image_shape

    out_h = (H - filter_h) // stride + 1
    out_w = (W - filter_w) // stride + 1

    img = np.zeros((C, H, W), dtype=cols.dtype)

    cols = cols.reshape(C, filter_h, filter_w, out_h, out_w)
    cols = cols.transpose(0, 3, 4, 1, 2)

    for y in range(out_h):
        for x in range(out_w):

            img[
                :,
                y * stride : y * stride + filter_h,
                x * stride : x * stride + filter_w,
            ] += cols[:, y, x]

    return img


@njit(cache=True, parallel=True, fastmath=True)
def _im2col_batch_numba(x, filter_h, filter_w, stride):

    B, C, H, W = x.shape

    out_h = (H - filter_h) // stride + 1
    out_w = (W - filter_w) // stride + 1

    cols = np.empty((B * out_h * out_w, C * filter_h * filter_w), dtype=np.float32)

    for row in prange(B * out_h * out_w):
        b = row // (out_h * out_w)
        local = row - b * out_h * out_w
        oh = local // out_w
        ow = local - oh * out_w

        col_idx = 0

        for c in range(C):
            for fh in range(filter_h):
                yy = oh * stride + fh

                for fw in range(filter_w):
                    xx = ow * stride + fw
                    cols[row, col_idx] = x[b, c, yy, xx]
                    col_idx += 1

    return cols


def im2col_batch(x, filter_h, filter_w, stride=1):

    if USE_NUMBA_ACCELERATION:
        x = x.astype(np.float32, copy=False)
        return _im2col_batch_numba(x, filter_h, filter_w, stride)

    B, C, H, W = x.shape

    out_h = (H - filter_h) // stride + 1
    out_w = (W - filter_w) // stride + 1

    shape = (B, C, out_h, out_w, filter_h, filter_w)

    strides = (
        x.strides[0],
        x.strides[1],
        stride * x.strides[2],
        stride * x.strides[3],
        x.strides[2],
        x.strides[3],
    )

    windows = as_strided(x, shape=shape, strides=strides)

    cols = windows.reshape(B * out_h * out_w, -1)

    return cols.astype(np.float32, copy=False)


@njit(cache=True, parallel=True, fastmath=True)
def _col2im_batch_numba(cols, B, C, H, W, filter_h, filter_w, stride, padding):

    H_padded = H + 2 * padding
    W_padded = W + 2 * padding

    out_h = (H_padded - filter_h) // stride + 1
    out_w = (W_padded - filter_w) // stride + 1

    x_padded = np.zeros((B, C, H_padded, W_padded), dtype=np.float32)

    for bc in prange(B * C):
        b = bc // C
        c = bc - b * C

        for oh in range(out_h):
            for ow in range(out_w):
                row = b * out_h * out_w + oh * out_w + ow

                for fh in range(filter_h):
                    yy = oh * stride + fh

                    for fw in range(filter_w):
                        xx = ow * stride + fw
                        col = c * filter_h * filter_w + fh * filter_w + fw
                        x_padded[b, c, yy, xx] += cols[row, col]

    if padding > 0:
        return x_padded[:, :, padding:H_padded - padding, padding:W_padded - padding]

    return x_padded


def col2im_batch(
    cols,
    x_shape,
    filter_h,
    filter_w,
    stride=1,
    padding=0
):

    B, C, H, W = x_shape

    if USE_NUMBA_ACCELERATION:
        cols = cols.astype(np.float32, copy=False)
        return _col2im_batch_numba(
            cols,
            B,
            C,
            H,
            W,
            filter_h,
            filter_w,
            stride,
            padding
        )

    H_padded = H + 2 * padding
    W_padded = W + 2 * padding

    out_h = (
        (H_padded - filter_h)
        // stride
        + 1
    )

    out_w = (
        (W_padded - filter_w)
        // stride
        + 1
    )

    cols = cols.reshape(
        B,
        out_h,
        out_w,
        C,
        filter_h,
        filter_w
    )

    x_padded = np.zeros(
        (B, C, H_padded, W_padded),
        dtype=np.float32
    )

    # =====================================
    # LESS TRANSPOSE OVERHEAD
    # =====================================

    cols = cols.transpose(0, 3, 4, 5, 1, 2)

    for y in range(filter_h):

        y_max = y + stride * out_h

        for x in range(filter_w):

            x_max = x + stride * out_w

            x_padded[
                :,
                :,
                y:y_max:stride,
                x:x_max:stride
            ] += cols[:, :, y, x]

    if padding > 0:

        return x_padded[
            :,
            :,
            padding:-padding,
            padding:-padding
        ]

    return x_padded
