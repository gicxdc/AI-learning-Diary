import numpy as np
from Foward import *
from Backward import *
from Other import *

def forwardPass(x,
                convFilter1, convBias1,
                convFilter2, convBias2,
                denseWeight, denseBias):

    #  Conv Block 1 
    convOut1, cache1 = convForward_im2col(x, convFilter1, convBias1, stride=1,padding=1)
    bnOut1, bnCache1 = batchnormForward(convOut1, gamma1, beta1)
    reluOut1 = reluFoward(bnOut1)
    poolOut1, poolCache1 = maxpoolForward_im2col(reluOut1, 2, 2)

    #  Conv Block 2 
    convOut2, cache2 = convForward_im2col(poolOut1, convFilter2, convBias2, stride=1,padding=1)
    bnOut2, bnCache2 = batchnormForward(convOut2, gamma2, beta2)
    reluOut2 = reluFoward(bnOut2)
    poolOut2, poolCache2 = maxpoolForward_im2col(reluOut2, 2, 2)

    #  Flatten 
    flat = poolOut2.reshape(-1, 1)
 
    
    #  Dense 
    logits, denseCache = denseForward(flat, denseWeight, denseBias)
    probs = softmaxFoward(logits)

    cache = (x,
             convOut1,bnOut1, reluOut1, poolOut1, poolCache1,
             convOut2,bnOut2, reluOut2, poolOut2, poolCache2,
             flat,
             cache1, cache2,bnCache1,bnCache2,
             denseCache )

    return probs, cache

def backwardPass(probs, label, cache,
                 convFilter1, convFilter2, denseWeight):

    (x,
     convOut1,bnOut1, reluOut1, poolOut1, poolCache1,
     convOut2,bnOut2, reluOut2, poolOut2, poolCache2,
     flat,
     cache1, cache2,bnCache1,bnCache2,
     denseCache ) = cache

    #  Softmax 
    dout = probs.copy()
    dout[label] -= 1

    #  Dense 
    dflat, ddenseWeight, ddenseBias = denseBackward(
        dout, flat, denseWeight, denseCache [2]
    )

    dpoolOut2 = dflat.reshape(poolOut2.shape)

    dreluOut2 = maxpoolBackward_im2col(dpoolOut2, poolCache2)

    dbnOut2 = reluBackward(dreluOut2, bnOut2)

    dconvOut2, dgamma2, dbeta2 = batchnormBackward(dbnOut2, bnCache2)

    dpoolOut1, dconvFilter2, dconvBias2 = convBackward_im2col(dconvOut2, cache2)

    dreluOut1 = maxpoolBackward_im2col(dpoolOut1, poolCache1)

    dbnOut1 = reluBackward(dreluOut1, bnOut1)

    dconvOut1, dgamma1, dbeta1 = batchnormBackward(dbnOut1, bnCache1)

    dx, dconvFilter1, dconvBias1 = convBackward_im2col(dconvOut1, cache1)

    return dconvFilter1, dconvBias1, dconvFilter2, dconvBias2, ddenseWeight, ddenseBias, dgamma1, dbeta1,dgamma2, dbeta2

def print_cache_shapes(cache):
    (
        image,
        con1, relu1, pool1, poolCache1,
        con2, relu2, pool2, poolCache2,
        flat,
        cache1, cache2,
        denseCache 
    ) = cache

    print("\n CNN CACHE SHAPES ")

    print("image      :", image.shape)

    print("\n--- Conv Block 1 ---")
    print("con1       :", con1.shape)
    print("relu1      :", relu1.shape)
    print("pool1      :", pool1.shape)

    print("\n--- Conv Block 2 ---")
    print("con2       :", con2.shape)
    print("relu2      :", relu2.shape)
    print("pool2      :", pool2.shape)

    print("\n--- Dense ---")
    print("flat       :", flat.shape)

    print("\n--- Params ---")
    print("cache1 (im2col conv1):", cache1[1].shape)  # X_col conv1
    print("cache2 (im2col conv2):", cache2[1].shape)  # X_col conv2

    print("\n==\n")

x_train = read_idx(r'D:\TeraBoxDownload\archive\train-images.idx3-ubyte')
y_train = read_idx(r'D:\TeraBoxDownload\archive\train-labels.idx1-ubyte')




print(x_train.shape)  # (60000, 28, 28)
print(y_train.shape)  # (60000,)

input_size = 32 * 7 * 7

#  Conv Layer 1 
convFilter1 = np.random.randn(16, 1, 3, 3) * 0.1
convBias1 = np.random.randn(16, 1) * 0.1

#  Conv Layer 2 
convFilter2 = np.random.randn(32, 16, 3, 3) * 0.1
convBias2 = np.random.randn(32, 1) * 0.1

#  Den se Layer 
denseWeight  = np.random.randn(10, input_size) * 0.1
denseBias = np.random.randn(10, 1) * 0.1


batchLoss=0

#  Conv1 
mconvFilter1 = np.zeros_like(convFilter1)
vconvFilter1 = np.zeros_like(convFilter1)
mconvBias1 = np.zeros_like(convBias1)
vconvBias1 = np.zeros_like(convBias1)

#  Conv2 
mconvFilter2 = np.zeros_like(convFilter2)
vconvFilter2 = np.zeros_like(convFilter2)
mconvBias2 = np.zeros_like(convBias2)
vconvBias2 = np.zeros_like(convBias2)

#  Dense 
mdenseWeight = np.zeros_like(denseWeight)
vdenseWeight = np.zeros_like(denseWeight)
mdenseBias = np.zeros_like(denseBias)
vdenseBias = np.zeros_like(denseBias)

gamma1 = np.ones((16,1,1))
beta1  = np.zeros((16,1,1))

gamma2 = np.ones((32,1,1))
beta2  = np.zeros((32,1,1))

#  Conv1 BatchNorm Adam states 
mg1 = np.zeros_like(gamma1)
vg1 = np.zeros_like(gamma1)

mb1 = np.zeros_like(beta1)
vb1 = np.zeros_like(beta1)

#  Conv2 BatchNorm Adam states 
mg2 = np.zeros_like(gamma2)
vg2 = np.zeros_like(gamma2)

mb2 = np.zeros_like(beta2)
vb2 = np.zeros_like(beta2)

epochs = 10
batchSize = 32
lr = 0.0003
t = 0
bestLoss = float("inf")

for epoch in range(epochs):

    total_loss = 0
    correct = 0

    perm = np.random.permutation(len(x_train))
    x_train = x_train[perm]
    y_train = y_train[perm]

    for i in range(0, len(x_train), batchSize):

        dconvFilter1_sum = np.zeros_like(convFilter1)
        dconvFilter2_sum = np.zeros_like(convFilter2)
        ddenseWeight_sum = np.zeros_like(denseWeight)

        dconvBias1_sum = np.zeros_like(convBias1)
        dconvBias2_sum = np.zeros_like(convBias2)
        ddenseBias_sum = np.zeros_like(denseBias)

        dgamma1_sum = np.zeros_like(gamma1)
        dbeta1_sum  = np.zeros_like(beta1)

        dgamma2_sum = np.zeros_like(gamma2)
        dbeta2_sum  = np.zeros_like(beta2)

        batch_loss = 0
        batch_correct = 0

        batch_end = min(i + batchSize, len(x_train))
        batch_len = batch_end - i

        for j in range(i, batch_end):

            x = x_train[j].reshape(1, 28, 28) / 255.0
            y = y_train[j]

            #  forward 
            probs, cache = forwardPass(x, convFilter1, convBias1, convFilter2, convBias2, denseWeight, denseBias)

            loss = cross_entropy(probs, y)
            batch_loss += loss

            pred = np.argmax(probs)
            if pred == y:
                batch_correct += 1

            #  backward 
            dconvFilter1, dconvBias1, dconvFilter2, dconvBias2, ddenseWeight, ddenseBias, dgamma1, dbeta1,dgamma2, dbeta2 = backwardPass(
                probs, y, cache, convFilter1, convFilter2, denseWeight
            )

            dconvFilter1_sum += dconvFilter1
            dconvFilter2_sum += dconvFilter2
            ddenseWeight_sum += ddenseWeight

            dconvBias1_sum += dconvBias1
            dconvBias2_sum += dconvBias2
            ddenseBias_sum += ddenseBias

            dgamma1_sum += dgamma1
            dbeta1_sum  += dbeta1

            dgamma2_sum += dgamma2
            dbeta2_sum  += dbeta2

        #  average grads 
        dconvFilter1_sum /= batch_len
        dconvFilter2_sum /= batch_len
        ddenseWeight_sum /= batch_len

        dconvBias1_sum /= batch_len
        dconvBias2_sum /= batch_len
        ddenseBias_sum /= batch_len

        #  Adam step 
        t += 1

        convFilter1, mconvFilter1, vconvFilter1 = adamOptimize(convFilter1, dconvFilter1_sum, mconvFilter1, vconvFilter1, t, lr, 0.9, 0.999, 1e-8)
        convBias1, mconvBias1, vconvBias1 = adamOptimize(convBias1, dconvBias1_sum, mconvBias1, vconvBias1, t, lr, 0.9, 0.999, 1e-8)

        convFilter2, mconvFilter2, vconvFilter2 = adamOptimize(convFilter2, dconvFilter2_sum, mconvFilter2, vconvFilter2, t, lr, 0.9, 0.999, 1e-8)
        convBias2, mconvBias2, vconvBias2 = adamOptimize(convBias2, dconvBias2_sum, mconvBias2, vconvBias2, t, lr, 0.9, 0.999, 1e-8)

        denseWeight, mdenseWeight, vdenseWeight = adamOptimize(denseWeight, ddenseWeight_sum, mdenseWeight, vdenseWeight, t, lr, 0.9, 0.999, 1e-8)
        denseBias, mdenseBias, vdenseBias = adamOptimize(denseBias, ddenseBias_sum, mdenseBias, vdenseBias, t, lr, 0.9, 0.999, 1e-8)

        gamma1, mg1, vg1 = adamOptimize(gamma1, dgamma1_sum, mg1, vg1, t, lr, 0.9, 0.999, 1e-8)
        beta1,  mb1, vb1 = adamOptimize(beta1,  dbeta1_sum,  mb1, vb1, t, lr, 0.9, 0.999, 1e-8)

        gamma2, mg2, vg2 = adamOptimize(gamma2, dgamma2_sum, mg2, vg2, t, lr, 0.9, 0.999, 1e-8)
        beta2,  mb2, vb2 = adamOptimize(beta2,  dbeta2_sum,  mb2, vb2, t, lr, 0.9, 0.999, 1e-8)

        total_loss += batch_loss
        correct += batch_correct

        if i %320==0:
            print(f"Epoch {epoch} | Step {i} | Loss {batch_loss / batch_len:.4f}")

    epoch_loss = total_loss / len(x_train)
    epoch_acc = correct / len(x_train)

    print("\n")
    print(f"Epoch {epoch}")
    print("Loss:", epoch_loss)
    print("Accuracy:", epoch_acc)
    print("\n")

    #  save best model 
    if epoch_loss < bestLoss:
        bestLoss = epoch_loss

        np.savez("MNIST_model.npz",
            convFilter1=convFilter1, convBias1=convBias1,
            convFilter2=convFilter2, convBias2=convBias2,
            denseWeight=denseWeight, denseBias=denseBias,
            mconvFilter1=mconvFilter1, vconvFilter1=vconvFilter1,
            mconvFilter2=mconvFilter2, vconvFilter2=vconvFilter2,
            mdenseWeight=mdenseWeight, vdenseWeight=vdenseWeight,
            mconvBias1=mconvBias1, vconvBias1=vconvBias1,
            mconvBias2=mconvBias2, vconvBias2=vconvBias2,
            mdenseBias=mdenseBias, vdenseBias=vdenseBias,
            t=t,
            bestLoss=bestLoss
        )




