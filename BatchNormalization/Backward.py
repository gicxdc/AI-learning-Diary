import numpy as np
from Other import *


def softmaxBackward(probs, label):
    dx = probs.copy()
    dx[label] -= 1
    return dx


def denseBackward(dout, input, weight, bias):
    # dW: (out, in)
    dW = dout @ input.T

    # db: (out, 1)
    db = np.sum(dout, axis=1, keepdims=True)

    # dx: (in, 1)
    dx = weight.T @ dout

    return dx, dW, db


def flattenBackward(dout, org_shape):
    return dout.reshape(org_shape)


def maxpoolBackward_im2col(dout, cache):
    image_shape = cache["image_shape"]
    image_col = cache["image_col"]
    max_idx = cache["max_idx"]
    size = cache["size"]
    stride = cache["stride"]

    imageChannel, imageHeight, imageWidth = image_shape
    dout = dout.reshape(imageChannel, -1)

    # create gradient in col space
    dX_col = np.zeros_like(image_col)

    c_idx = np.arange(imageChannel)[:, None]
    w_idx = np.arange(dout.shape[1])[None, :]

    dX_col[c_idx, max_idx, w_idx] = dout[c_idx, w_idx]

    # reshape for col2im
    dX_col = dX_col.reshape(imageChannel * size * size, -1)

    # scatter back to image space
    dx = col2im(dX_col, image_shape, size, size, stride)

    return dx


def reluBackward(dout, input):
    dx = dout.copy()
    dx[input <= 0] = 0
    return dx


def convBackward_im2col(dout, cache):
    image, imagePadded, filters, bias, X_col, stride, padding = cache

    filterNumber, filterChannel, filterHeight, filterWidth = filters.shape
    imageChannel, imageHeight, imageWidth = imagePadded.shape

    dout_col = dout.reshape(filterNumber, -1)  # (C_out, outH*outW)

    #  gradients 
    W_col = filters.reshape(filterNumber, -1)

    # dW
    dW = dout_col @ X_col.T
    dW = dW.reshape(filters.shape)

    # db
    db = np.sum(dout_col, axis=1).reshape(-1, 1)

    # dX_col
    dX_col = W_col.T @ dout_col  # (C_in*kH*kW, outH*outW)

    # dX
    dX_padded = col2im(dX_col, imagePadded.shape, filterHeight, filterWidth, stride)

    if padding > 0:
        dX = dX_padded[:, padding:-padding, padding:-padding]
    else:
        dX = dX_padded

    return dX, dW, db


def batchnormBackward(dout, cache):

    x, xhat, mean, var, gamma, beta, eps = cache
    B, C, H, W = x.shape

    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)
    dgamma = np.sum(dout * xhat, axis=(0, 2, 3), keepdims=True)

    dxhat = dout * gamma

    dvar = np.sum(
        dxhat * (x - mean) * -0.5 * (var + eps) ** (-1.5), axis=(0, 2, 3), keepdims=True
    )
    dmean = np.sum(
        dxhat * -1 / np.sqrt(var + eps), axis=(0, 2, 3), keepdims=True
    ) + dvar * np.mean(-2 * (x - mean), axis=(0, 2, 3), keepdims=True)

    dx = (
        dxhat / np.sqrt(var + eps)
        + dvar * 2 * (x - mean) / (B * H * W)
        + dmean / (B * H * W)
    )

    return dx, dgamma, dbeta


def dropout_backward(dout, cache):
    mask = cache
    dx = dout * mask
    return dx
