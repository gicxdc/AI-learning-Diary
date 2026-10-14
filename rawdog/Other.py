import numpy as np

def softmax(z):
    z = z - np.max(z)       
    exp = np.exp(z)
    return exp / np.sum(exp)

def categoricalCrossEntropy(probs, label):
    return float(-np.log(probs[label] + 1e-8))

def adamOptimize(param,dparam,m,v,t,learnRate,beta1,beta2,eps):
    m = beta1 * m + (1 - beta1) * dparam
    v = beta2 * v + (1 - beta2) * (dparam ** 2)

    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    param -= learnRate * m_hat / (np.sqrt(v_hat) + eps)

    return param,m,v


def im2col(image, filterHeight,filterWidth, stride=1):
    imgChanel, imgHeight, imgWidth = image.shape
    

    out_h = (imgHeight - filterHeight) // stride + 1
    out_w = (imgWidth - filterWidth) // stride + 1

    cols = np.zeros((imgChanel * filterHeight * filterWidth, out_h * out_w))

    col_idx = 0
    for y in range(out_h):
        for x in range(out_w):
            patch = image[:, 
                          y*stride:y*stride+filterHeight,
                          x*stride:x*stride+filterWidth]
            cols[:, col_idx] = patch.reshape(-1)
            col_idx += 1

    return cols  # shape: (C*fh*fw, out_h*out_w)

def col2im(cols, image_shape, filter_h, filter_w, stride=1):
    imgChanel, imgHeight, imgWidth= image_shape
    out_h = (imgHeight - filter_h)//stride + 1
    out_w = (imgWidth - filter_w)//stride + 1

    img = np.zeros((imgChanel,imgHeight,imgWidth),dtype=cols.dtype)

    col_idx = 0
    for y in range(out_h):
        for x in range(out_w):
            patch = cols[:, col_idx].reshape(imgChanel, filter_h, filter_w)

            img[:, 
                y*stride:y*stride+filter_h,
                x*stride:x*stride+filter_w] += patch

            col_idx += 1

    return img