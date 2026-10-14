import numpy as np
import struct


def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(dims))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
        return data

def cross_entropy(probs, label):
    return -np.log(probs[label, 0] + 1e-9)


def im2col(image, filter_h, filter_w, stride=1):
    channel, height, width = image.shape

    out_h = (height - filter_h) // stride + 1
    out_w = (width - filter_w) // stride + 1

    patches = []

    for y in range(out_h):
        for x in range(out_w):
            patch = image[:, 
                          y*stride:y*stride+filter_h,
                          x*stride:x*stride+filter_w]
            patches.append(patch.reshape(-1))

    return np.stack(patches, axis=1) # shape: (C*fh*fw, out_h*out_w)

def col2im(cols, image_shape, filter_h, filter_w, stride=1):
    channel, height, width = image_shape

    out_h = (height - filter_h) // stride + 1
    out_w = (width - filter_w) // stride + 1

    img = np.zeros((channel, height, width), dtype=cols.dtype)

    cols = cols.reshape(channel, filter_h, filter_w, out_h, out_w)
    cols = cols.transpose(0, 3, 4, 1, 2)  # (C, out_h, out_w, fh, fw)

    for y in range(out_h):
        for x in range(out_w):
            img[:, 
                y*stride:y*stride+filter_h,
                x*stride:x*stride+filter_w] += cols[:, y, x]

    return img

def adamOptimize(param,dparam,m,v,t,learnRate,beta1,beta2,eps):
    m = beta1 * m + (1 - beta1) * dparam
    v = beta2 * v + (1 - beta2) * (dparam ** 2)

    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    param -= learnRate * m_hat / (np.sqrt(v_hat) + eps)

    return param,m,v

def pad_image(x, pad):
    if pad == 0:
        return x
    return np.pad(
        x,
        ((0, 0), (pad, pad), (pad, pad)),
        mode='constant'
    )