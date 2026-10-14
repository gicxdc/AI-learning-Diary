import numpy as np
from Other import *


def relu_backward(dout, input):
    dx = dout.copy()
    dx[input <= 0] = 0
    return dx

def backward_dense(dout, input, weight, bias):
 # gradient wrt weights
    dW = dout @ input.T

    # gradient wrt bias
    db = np.sum(dout, axis=1, keepdims=True)

    # gradient wrt input (to pass backward)
    dx = weight.T @ dout

    return dx, dW, db

def backward_flatten(dout,org_shape):
       return dout.reshape(org_shape)

def backward_con(gradient, convInput, filters, stride):

    (filterNumber, filterChannel, filterHeight, filterWidth) = filters.shape
    (_, convHeight, convWidth) = convInput.shape

    dout = np.zeros_like(convInput)
    dfilter = np.zeros_like(filters)
    dbias = np.zeros((filterNumber,))

    for curr_f in range(filterNumber):
        out_y = 0

        for curr_y in range(0, convHeight - filterHeight+ 1, stride):
            out_x = 0

            for curr_x in range(0, convWidth - filterWidth + 1, stride):

                region = convInput[:, curr_y:curr_y+filterHeight, curr_x:curr_x+filterWidth]
                grad = gradient[curr_f, out_y, out_x]

                dfilter[curr_f] += grad * region
                dout[:, curr_y:curr_y+filterHeight, curr_x:curr_x+filterWidth] += grad * filters[curr_f]

                out_x += 1

            out_y += 1

        dbias[curr_f] = np.sum(gradient[curr_f])

    return dout, dfilter, dbias

def maxpoolBackward(dpool, org, filter, stride):
    (orgChannel,orgHeight,orgWidth)=org.shape
    dout=np.zeros(org.shape)
    
    for curr_c in range(orgChannel):
        curr_y = out_y = 0

        while curr_y + filter <= orgHeight:
            curr_x = out_x = 0

            while curr_x + filter <= orgWidth:
                window = org[curr_c, curr_y:curr_y+filter, curr_x:curr_x+filter]

                (a, b) = nanargmax(window)

                dout[curr_c, curr_y+a, curr_x+b] += dpool[curr_c, out_y, out_x]
                
                curr_x += stride
                out_x += 1
            
            curr_y += stride
            out_y += 1
    
    return dout


def nanargmax(arr):
    idx = np.nanargmax(arr)
    idxs = np.unravel_index(idx, arr.shape)
    return idxs    

def conv_im2col_backward(dout, X_col, image, filters, stride=1):
    filterNumber, filterChannel, filterHeight, filterWidth = filters.shape

    dout_reshaped = dout.reshape(filterNumber, -1)

    # gradients
    dW = (dout_reshaped @ X_col.T).reshape(filters.shape)
    db = np.sum(dout_reshaped, axis=1,keepdims=True)

    dX_col = filters.reshape(filterNumber, -1).T @ dout_reshaped

    dX = col2im(dX_col, image.shape, filterHeight, filterWidth, stride)

    return dX, dW, db

def maxpoolBackward_vector(dout, mask, size=2):
    channel, out_h, out_w = dout.shape

    # expand dout to match mask
    dout_expanded = dout[:, :, :, None, None]

    # distribute gradients only to max locations
    dmax = mask * dout_expanded

    # reshape back
    dmax = dmax.transpose(0, 1, 3, 2, 4)
    dx = dmax.reshape(channel, out_h * size, out_w * size)

    return dx