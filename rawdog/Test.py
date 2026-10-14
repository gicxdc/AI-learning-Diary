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

print(x_train.shape)  # (60000, 28, 28)
print(y_train.shape)  # (60000,)


def forwardPass(image, filters, weight,convBias, denseBias):
    conOut = conv(image, filters, convBias)
    reluOut = relu(conOut)
    poolOut = maxpool(reluOut, 2, 2)
    flattenOut = flatten(poolOut)
    out = dense(flattenOut, weight, denseBias)
    probs = softmax(out)

    outStorage=(conOut,reluOut,poolOut,flattenOut)

    return probs,outStorage

def backwardPass(image,filters,weight,denseBias,probs,outStorage,label):
    conOut, reluOut, poolOut, flattenOut = outStorage

    dout = probs.copy()
    dout[label] -= 1 

    dfc, dW, dbDense = backward_dense(dout, flattenOut, weight, denseBias)
    dpooled = backward_flatten(dfc, poolOut.shape)
    drelu = maxpoolBackward(dpooled, reluOut, 2, 2)
    dconv = relu_backward(drelu, conOut)
    dX, dF, dbConv = backward_con(dconv, image, filters, 1)

    return dW, dbDense, dF, dbConv






filters = np.random.randn(8, 1, 3, 3) * 0.1
convBias = np.random.randn(8) * 0.1

num_classes = 10
input_size = 8 * 13 * 13

weight = np.random.randn(num_classes, input_size) * 0.1
denseBias = np.random.randn(num_classes, 1) * 0.1

t=0

learnRate=0.0005
beta1 = 0.9
beta2 = 0.999
eps = 1e-8

mWeight= np.zeros_like(weight)
vWeight= np.zeros_like(weight)

mbDense = np.zeros_like(denseBias)
vbDense = np.zeros_like(denseBias)

mFilter = np.zeros_like(filters)
vFilter = np.zeros_like(filters)

mbConv = np.zeros_like(convBias)
vbConv = np.zeros_like(convBias)

epochs=10
batchSize=32
bestLoss=10

for epoch in range(epochs):
    total_loss = 0
    correct = 0
    perm = np.random.permutation(len(x_train))
    x_shuffled = x_train[perm]
    y_shuffled = y_train[perm]
  
    
    for i in range(0, len(x_train), batchSize):

        dW_sum = np.zeros_like(weight)
        db_sum = np.zeros_like(denseBias)
        dF_sum = np.zeros_like(filters)
        db_conv_sum = np.zeros_like(convBias)
        batchLoss=0

        for j in range(i, min(i + batchSize, len(x_train))):

            image = x_shuffled[j] / 255.0
            image = image.reshape(1, 28, 28)
            label = y_shuffled[j]

            probs, outStorage = forwardPass(image, filters, weight, convBias, denseBias)

            loss = categoricalCrossEntropy(probs, label)
            batchLoss += loss

            pred = np.argmax(probs)
            if pred == label:
                correct += 1

            dW, db, dF, db_conv = backwardPass(
                image, filters, weight, denseBias,
                probs, outStorage, label
            )

            dW_sum += dW
            db_sum += db
            dF_sum += dF
            db_conv_sum += db_conv

        batch_len = min(batchSize, len(x_train) - i)

        dW_avg = dW_sum / batch_len
        db_avg = db_sum / batch_len
        dF_avg = dF_sum / batch_len
        db_conv_avg = db_conv_sum / batch_len
        total_loss += batchLoss
        t += 1

        weight, mWeight, vWeight = adamOptimize(weight, dW_avg, mWeight, vWeight, t, learnRate, beta1, beta2, eps)

        denseBias, mbDense, vbDense = adamOptimize(denseBias, db_avg, mbDense, vbDense, t, learnRate, beta1, beta2, eps)

        filters, mFilter, vFilter = adamOptimize(filters, dF_avg, mFilter, vFilter, t, learnRate, beta1, beta2, eps)

        convBias, mbConv, vbConv = adamOptimize(convBias, db_conv_avg, mbConv, vbConv, t, learnRate, beta1, beta2, eps)
    
        if i % 320 == 0 and i > 0:
                print(f"Epoch {epoch}, Step {i}, Batch Loss: {batchLoss / batch_len}")
    

    epochLoss = total_loss / len(x_train)
    if epochLoss < bestLoss:
        bestLoss = epochLoss
        np.savez("model.npz",
            filters=filters,
            convBias=convBias,
            weight=weight,
            denseBias=denseBias)

    print("\n======================")
    print(f"Epoch {epoch} done")
    print("Avg loss:", total_loss / len(x_train))
    print("Accuracy:", correct / len(x_train))
    print("======================\n")


    
