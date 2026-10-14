import numpy as np
import struct
from Other import *

def convForward_im2col(image, filters, bias, stride=1,padding=0):
    filterNumber,filterChanel,filterHeight,filterWidth=filters.shape
    imageChannel,imageHeight,imageWidth=image.shape

    assert filterChanel==imageChannel, "channel mismatch"
    imagePadded=pad_image(image,padding)

    image_col = im2col(imagePadded, filterHeight, filterWidth, stride)
    filters_col = filters.reshape(filterNumber, -1)

    out = filters_col @ image_col + bias.reshape(-1, 1)
    _, paddedH, paddedW = imagePadded.shape

    outChannel= filterNumber
    outHeight = (paddedH - filterHeight) // stride + 1
    outWidth  = (paddedW - filterWidth) // stride + 1
        
        
    out = out.reshape(outChannel, outHeight, outWidth)

    cache = (image,imagePadded, filters, bias, image_col, stride,padding)
    return out, cache

def reluFoward(x):
    return np.maximum(0, x)

def maxpoolForward_im2col(image, size=2, stride=2):
    imageChannel,imageHeight,imageWidth=image.shape

    image_col = im2col(image, size, size, stride)
    image_col = image_col.reshape(imageChannel, size*size, -1)

    out = np.max(image_col, axis=1)
    max_idx = np.argmax(image_col, axis=1)

    outChannel= imageChannel
    outHeight = (imageHeight - size) // stride + 1
    outWidth  = (imageWidth - size) // stride + 1

    out=out.reshape(outChannel,outHeight,outWidth)

    cache = {
        "image_shape": image.shape,
        "image_col": image_col,
        "max_idx": max_idx,
        "size": size,
        "stride": stride
    }
    return out, cache

def flattenFoward(image):
    imageChannel,imageHeight,imageWidth=image.shape
    return image.reshape(imageChannel*imageHeight*imageWidth,1)


def denseForward(image, weight, bias):
    out = weight @ image + bias.reshape(-1, 1)
    cache = (image, weight, bias)
    return out, cache

def softmaxFoward(x):
    x = x - np.max(x,axis=0, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp)

def batchnormForward(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=(1, 2), keepdims=True)
    var  = np.var(x, axis=(1, 2), keepdims=True)

    #  normalize 
    x_hat = (x - mean) / np.sqrt(var + eps)

    #  scale + shift 
    out = gamma * x_hat + beta

    cache = (x, x_hat, mean, var, gamma, beta, eps)
    return out, cache