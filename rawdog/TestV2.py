import numpy as np
import struct

from Foward import *
from Backward import *
from Other import *


def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(dims))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
        return data

x_train = read_idx(r'D:\TeraBoxDownload\archive\train-images.idx3-ubyte')
y_train = read_idx(r'D:\TeraBoxDownload\archive\train-labels.idx1-ubyte')
x_test = read_idx(r'D:\TeraBoxDownload\archive\t10k-images.idx3-ubyte')
y_test = read_idx(r'D:\TeraBoxDownload\archive\t10k-labels.idx1-ubyte')

def forwardPass(image, filters1, filters2, weight, convBias1, convBias2, denseBias):

    #  Conv block 1 
    con1, X1 = conv_im2col(image, filters1, convBias1,1)
    relu1 = relu(con1)
    pool1, mask1 = maxpool_vector(relu1, 2, 2)

    #  Conv block 2 
    con2, X2 = conv_im2col(pool1, filters2, convBias2,1)
    relu2 = relu(con2)
    pool2, mask2 = maxpool_vector(relu2, 2, 2)

    #  Dense 
    flat = flatten(pool2)
    out = dense(flat, weight, denseBias)
    probs = softmax(out)

    outStorage = (con1, relu1, pool1, con2, relu2, pool2, flat, X1, X2, mask1, mask2)

    return probs, outStorage

def backwardPass(image, filters1, filters2, weight, denseBias, probs, outStorage, label):

    con1, relu1, pool1, con2, relu2, pool2, flat, X1, X2, mask1, mask2 = outStorage

    #  Softmax 
    dout = probs.copy()
    dout[label] -= 1

    #  Dense 
    dflat, dW, dbDense = backward_dense(dout, flat, weight, denseBias)

    #  Unflatten 
    dpool2 = backward_flatten(dflat, pool2.shape)

    #  Conv block 2 
    drelu2 = maxpoolBackward_vector(dpool2, mask2, 2, 2)
    dconv2 = relu_backward(drelu2, con2)

    dpool1, dF2, dbConv2 = conv_im2col_backward(
        dconv2, X2, pool1, filters2,1
    )

    #  Conv block 1 
    drelu1 = maxpoolBackward_vector(dpool1, mask1, 2)
    dconv1 = relu_backward(drelu1, con1)

    dX, dF1, dbConv1 = conv_im2col_backward(
        dconv1, X1, image, filters1,1
    )

    return dW, dbDense, dF1, dbConv1, dF2, dbConv2


print(x_train.shape)  # (60000, 28, 28)
print(y_train.shape)  # (60000,)

# input_size = 16 * 6 * 6
input_size=16*5*5

filters1 = np.random.randn(8, 1, 3, 3) * 0.1
convBias1 = np.random.randn(8) * 0.1

filters2 = np.random.randn(16, 8, 3, 3) * 0.1
convBias2 = np.random.randn(16) * 0.1


weight = np.random.randn(10, input_size) * 0.1
denseBias = np.random.randn(10, 1) * 0.1

batchLoss=0

dW_sum=np.zeros_like(weight)
db_sum=np.zeros_like(denseBias)
dF1_sum=np.zeros_like(filters1)
dbConv1_sum=np.zeros_like(convBias1)
dF2_sum=np.zeros_like(filters2)
dbConv2_sum=np.zeros_like(convBias2)

mWeight= np.zeros_like(weight)
vWeight= np.zeros_like(weight)

mbDense = np.zeros_like(denseBias)
vbDense = np.zeros_like(denseBias)

mFilter1 = np.zeros_like(filters1)
vFilter1 = np.zeros_like(filters1)

mbConv1 = np.zeros_like(convBias1)
vbConv1 = np.zeros_like(convBias1)

mFilter2 = np.zeros_like(filters2)
vFilter2 = np.zeros_like(filters2)

mbConv2 = np.zeros_like(convBias2)
vbConv2 = np.zeros_like(convBias2)

epochs = 10
batchSize = 32
learnRate = 0.0003

t = 0
bestLoss = 1e9

for epoch in range(epochs):

    total_loss = 0
    correct = 0

    # shuffle dataset
    perm = np.random.permutation(len(x_train))
    x_train = x_train[perm]
    y_train = y_train[perm]

    for i in range(0, len(x_train), batchSize):

        # reset batch gradients
        dW_sum = np.zeros_like(weight)
        dbDense_sum = np.zeros_like(denseBias)
        dF1_sum = np.zeros_like(filters1)
        db1_sum = np.zeros_like(convBias1)
        dF2_sum = np.zeros_like(filters2)
        db2_sum = np.zeros_like(convBias2)

        batch_loss = 0

        batch_end = min(i + batchSize, len(x_train))
        batch_len = batch_end - i

        for j in range(i, batch_end):

            #  preprocess 
            image = x_train[j] / 255.0
            image = image.reshape(1, 28, 28)

            label = y_train[j]

            #  forward 
            probs, cache = forwardPass(
                image,
                filters1, filters2,
                weight,
                convBias1, convBias2,
                denseBias
            )

            loss = categoricalCrossEntropy(probs, label)
            batch_loss += loss

            pred = np.argmax(probs)
            if pred == label:
                correct += 1

            #  backward 
            dW, dbDense, dF1, db1, dF2, db2 = backwardPass(
                image,
                filters1, filters2,
                weight,
                denseBias,
                probs,
                cache,
                label
            )

            #  accumulate grads 
            dW_sum += dW
            dbDense_sum += dbDense
            dF1_sum += dF1
            db1_sum += db1
            dF2_sum += dF2
            db2_sum += db2

        #  average gradients 
        dW_avg = dW_sum / batch_len
        dbDense_avg = dbDense_sum / batch_len
        dF1_avg = dF1_sum / batch_len
        db1_avg = db1_sum / batch_len
        dF2_avg = dF2_sum / batch_len
        db2_avg = db2_sum / batch_len

        t += 1

        #  Adam updates 
        weight, mWeight, vWeight = adamOptimize(weight, dW_avg, mWeight, vWeight, t, learnRate, 0.9, 0.999, 1e-8)

        denseBias, mbDense, vbDense = adamOptimize(denseBias, dbDense_avg, mbDense, vbDense, t, learnRate, 0.9, 0.999, 1e-8)

        filters1, mFilter1, vFilter1 = adamOptimize(filters1, dF1_avg, mFilter1, vFilter1, t, learnRate, 0.9, 0.999, 1e-8)

        convBias1, mbConv1, vbConv1 = adamOptimize(convBias1, db1_avg, mbConv1, vbConv1, t, learnRate, 0.9, 0.999, 1e-8)

        filters2, mFilter2, vFilter2 = adamOptimize(filters2, dF2_avg, mFilter2, vFilter2, t, learnRate, 0.9, 0.999, 1e-8)

        convBias2, mbConv2, vbConv2 = adamOptimize(convBias2, db2_avg, mbConv2, vbConv2, t, learnRate, 0.9, 0.999, 1e-8)

        total_loss += batch_loss

        if i % 320 == 0:
            print(f"Epoch {epoch} | Step {i} | Loss {batch_loss / batch_len:.4f}")

    epoch_loss = total_loss / len(x_train)
    epoch_acc = correct / len(x_train)



    print("\n====================")
    print(f"Epoch {epoch}")
    print("Loss:", epoch_loss)
    print("Accuracy:", epoch_acc)
    print("====================\n")