import numpy as np
from Other import *
def conv(image,filters,bias,stride=1):
    (filterNumber,filterChanel,filterDimension,_)=filters.shape
    imgChanel,imgHeight, imgWidth=image.shape
    
    assert imgChanel==filterChanel, "filter and img must have the same chanel"

    #calculate out put
    outputWidth=int((imgWidth-filterDimension)/stride)+1
    outputHeight=int((imgHeight-filterDimension)/stride)+1
    
    out=np.zeros((filterNumber,outputHeight,outputWidth))
    
    #loop through each filter
    for curr_filter in range(filterNumber):
        curr_y=out_y=0

        while curr_y + filterDimension<=imgHeight:
            curr_x = out_x = 0
            while curr_x + filterDimension<=imgWidth:
               #extract the overlap region??
                temp = image[:,curr_y:curr_y+filterDimension, curr_x:curr_x+filterDimension]
                #calculate the  acutal result out=sum(filter)*extract reigion +bias
                out[curr_filter,out_y,out_x]=np.sum(filters[curr_filter]*temp)+bias[curr_filter]
               
                curr_x += stride
                out_x += 1
            curr_y += stride
            out_y += 1

    return out

def maxpool(image,filterDimenision=2,stride=2):
    imgChannel,imageHeight,imageWidth=image.shape

    #calculate output
    height=int((imageHeight-filterDimenision)/stride)+1
    width=int((imageWidth-filterDimenision)/stride)

    downSample=np.zeros((imgChannel,height,width))
    
    #loop through each chanel   
    for i in range (imgChannel):
        curr_y = out_y = 0

        while curr_y+filterDimenision<=imageHeight:
            curr_x = out_x = 0
            while curr_x+filterDimenision<=imageWidth:
                #extract the rmax in the reigion
                downSample[i, out_y, out_x] = np.max(image[i, curr_y:curr_y+filterDimenision, curr_x:curr_x+filterDimenision])
                
                curr_x += stride
                out_x += 1
            
            curr_y += stride
            out_y += 1
    return downSample

def relu(x):
    return np.maximum(0, x)
#convert to array
def flatten(image):
    imageChannel,imageHeight,imageWidth=image.shape
    return image.reshape(imageChannel*imageHeight*imageWidth,1)

def dense(image, weight, bias):
    return weight @ image + bias


def conv_im2col(image, filters, bias, stride=1):
    filterNumber, filterChannel, filterHeight, filterWidth = filters.shape
    imgChanel, imgHeight, imgWidth = image.shape

    assert imgChanel==filterChannel, "Channel mismatch"

    out_h = (imgHeight - filterHeight)//stride + 1
    out_w = (imgWidth - filterWidth)//stride + 1

    X_col = im2col(image, filterHeight, filterWidth, stride)  
    W_col = filters.reshape(filterNumber, -1)

    out = W_col @ X_col + bias.reshape(-1, 1)

    out = out.reshape(filterNumber, out_h, out_w)
    return out, X_col






def pool_im2col(x, size=2, stride=2):
    chanel, height, width = x.shape

    out_h = (height - size) // stride + 1
    out_w = (width - size) // stride + 1

    cols = []

    for y in range(out_h):
        for x_pos in range(out_w):
            patch = x[:, 
                      y*stride:y*stride+size,
                      x_pos*stride:x_pos*stride+size]
            cols.append(patch.reshape(-1))

    return np.array(cols).T   # (C*size*size, out_h*out_w)

def maxpool_im2col(x, size=2, stride=2):
    channel, height, width = x.shape

    X_col = pool_im2col(x, size, stride)

    # reshape into (C, size*size, num_windows)
    X_col = X_col.reshape(channel, size*size, -1)

    out = np.max(X_col, axis=1)   # (C, num_windows)

    # store argmax mask for backward
    max_idx = np.argmax(X_col, axis=1)

    mask = np.zeros_like(X_col)

    c_idx = np.arange(channel)[:, None]
    w_idx = np.arange(X_col.shape[2])[None, :]

    mask[c_idx, max_idx, w_idx] = 1

    out_h = (height - size) // stride + 1
    out_w = (width - size) // stride + 1

    return out.reshape(channel, out_h, out_w), mask, X_col.shape


def col2im_pool(cols, image_shape, size, stride):
    channel, height, width = image_shape
    out = np.zeros(image_shape)

    out_h = (height - size) // stride + 1
    out_w = (width - size) // stride + 1

    col_idx = 0

    for y in range(out_h):
        for x in range(out_w):
            patch = cols[:, col_idx].reshape(channel, size, size)

            out[:, 
                y*stride:y*stride+size,
                x*stride:x*stride+size] += patch

            col_idx += 1

    return out

def maxpoolBackward_im2col(dout, mask, x_shape, size=2, stride=2):
    channel, height, width = x_shape

    out_h = (height - size) // stride + 1
    out_w = (width - size) // stride + 1

    dout = dout.reshape(channel, -1)

    dX_col = mask * dout[:, None, :]

    # sum overlapping contributions
    dX_col = dX_col.reshape(channel * size * size, -1)

    dX = col2im_pool(dX_col, x_shape, size, stride)

    return dX




