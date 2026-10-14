import numpy as np
from Foward import *
from Backward import *
from Other import *
import matplotlib.pyplot as plt
import time


def forwardPass(
    x,
    convFilter1,
    convBias1,
    convFilter2,
    convBias2,
    denseWeight,
    denseBias,
    gamma1,
    beta1,
    gamma2,
    beta2,
    runningMean1,
    runningVar1,
    runningMean2,
    runningVar2,
):

    B = x.shape[0]

    #  Conv1 
    conv1_out, cache1_list = [], []

    for b in range(B):
        out, cache = convForward_im2col(x[b], convFilter1, convBias1, 1, 1)
        conv1_out.append(out)
        cache1_list.append(cache)

    conv1_out = np.stack(conv1_out, axis=0)
    bn1_out, bnCache1, runningMean1, runningVar1 = batchnormForward(
        conv1_out, gamma1, beta1, runningMean1, runningVar1
    )

    relu1 = reluFoward(bn1_out)

    pool1, poolCache1 = [], []
    for b in range(B):
        out, cache = maxpoolForward_im2col(relu1[b], 2, 2)
        pool1.append(out)
        poolCache1.append(cache)

    pool1 = np.stack(pool1)

    #  Conv2 
    conv2_out, cache2_list = [], []

    for b in range(B):
        out, cache = convForward_im2col(pool1[b], convFilter2, convBias2, 1, 1)
        conv2_out.append(out)
        cache2_list.append(cache)

    conv2_out = np.stack(conv2_out, axis=0)

    bn2_out, bnCache2, runningMean2, runningVar2 = batchnormForward(
        conv2_out, gamma2, beta2, runningMean2, runningVar2
    )

    relu2 = reluFoward(bn2_out)

    pool2, poolCache2 = [], []
    for b in range(B):
        out, cache = maxpoolForward_im2col(relu2[b], 2, 2)
        pool2.append(out)
        poolCache2.append(cache)

    pool2 = np.stack(pool2)

    #  Flatten 
    flat = pool2.reshape(B, -1).T

    #  Dense 
    logits, denseCache = denseForward(flat, denseWeight, denseBias)
    # dropout, dropCache = dropout_forward(logits)
    probs = softmaxFoward(logits)

    cache = (
        x,
        conv1_out,
        bn1_out,
        relu1,
        pool1,
        conv2_out,
        bn2_out,
        relu2,
        pool2,
        poolCache1,
        poolCache2,
        cache1_list,
        cache2_list,
        bnCache1,
        bnCache2,
        flat,
        denseCache,
    )

    return probs, cache, runningMean1, runningVar1, runningMean2, runningVar2


def backwardPass(probs, y_batch, cache, convFilter1, convFilter2, denseWeight):

    (
        x,
        conv1_out,
        bn1_out,
        relu1,
        pool1,
        conv2_out,
        bn2_out,
        relu2,
        pool2,
        poolCache1,
        poolCache2,
        cache1_list,
        cache2_list,
        bnCache1,
        bnCache2,
        flat,
        denseCache,
    ) = cache

    B = x.shape[0]

    #  softmax 
    dout = probs.copy()
    for i in range(B):
        dout[y_batch[i], i] -= 1
    dout /= B

    # dout = dropout_backward(dout, dropCache)
    #  dense 
    dflat, ddW, ddb = denseBackward(dout, flat, denseWeight, denseCache[2])
    dpool2 = dflat.T.reshape(pool2.shape)

    #  pool2 
    drelu2 = np.stack(
        [maxpoolBackward_im2col(dpool2[b], poolCache2[b]) for b in range(B)]
    )

    #  relu2 
    dbn2 = reluBackward(drelu2, bn2_out)

    #  BN2 
    dconv2, dgamma2, dbeta2 = batchnormBackward(dbn2, bnCache2)

    #  conv2 
    dconvFilter2 = np.zeros_like(convFilter2)
    dconvBias2 = np.zeros_like(convBias2)

    dpool1 = np.zeros_like(pool1)

    for b in range(B):
        dx, dw, db = convBackward_im2col(dconv2[b], cache2_list[b])
        dconvFilter2 += dw
        dconvBias2 += db
        dpool1[b] = dx

    # dconvFilter2 /= B
    # dconvBias2 /= B

    #  pool1 
    drelu1 = np.stack(
        [maxpoolBackward_im2col(dpool1[b], poolCache1[b]) for b in range(B)]
    )

    #  relu1 
    dbn1 = reluBackward(drelu1, bn1_out)

    #  BN1 
    dconv1, dgamma1, dbeta1 = batchnormBackward(dbn1, bnCache1)

    #  conv1 
    dconvFilter1 = np.zeros_like(convFilter1)
    dconvBias1 = np.zeros_like(convBias1)

    dx = np.zeros_like(x)

    for b in range(B):
        dxi, dw, db = convBackward_im2col(dconv1[b], cache1_list[b])
        dconvFilter1 += dw
        dconvBias1 += db
        dx[b] = dxi

    # dconvFilter1 /= B
    # dconvBias1 /= B

    return (
        dconvFilter1,
        dconvBias1,
        dconvFilter2,
        dconvBias2,
        ddW,
        ddb,
        dgamma1,
        dbeta1,
        dgamma2,
        dbeta2,
    )


def print_cache_shapes(cache):
    (
        image,
        con1,
        relu1,
        pool1,
        poolCache1,
        con2,
        relu2,
        pool2,
        poolCache2,
        flat,
        cache1,
        cache2,
        denseCache,
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


x_train = read_idx(r"D:\TeraBoxDownload\archive\train-images.idx3-ubyte")
y_train = read_idx(r"D:\TeraBoxDownload\archive\train-labels.idx1-ubyte")


print(x_train.shape)  # (60000, 28, 28)
print(y_train.shape)  # (60000,)

input_size = 32 * 7 * 7

#  Conv Layer 1 
convFilter1 = he_init((16, 1, 3, 3)) 
convBias1 = np.zeros((16,1))

#  Conv Layer 2 
convFilter2 = he_init((32, 16, 3, 3)) 
convBias2 = np.zeros((32,1))

#  Den se Layer 
denseWeight = he_init((10, input_size)) 
denseBias = np.zeros((10,1))


batchLoss = 0

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

gamma1 = np.ones((16, 1, 1))
beta1 = np.zeros((16, 1, 1))

gamma2 = np.ones((32, 1, 1))
beta2 = np.zeros((32, 1, 1))

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


# BatchNorm Running Statistics


runningMean1 = np.zeros((16, 1, 1))
runningVar1 = np.ones((16, 1, 1))

runningMean2 = np.zeros((32, 1, 1))
runningVar2 = np.ones((32, 1, 1))

epochs = 10
batchSize = 32
lr = 0.001
t = 0
bestLoss = float("inf")
weight_decay = 0.0005


train_losses = []
train_accs = []


# TRAINING LOOP


for epoch in range(epochs):
    epoch_start = time.time()
    # ==
    # Shuffle dataset
    # ==
    perm = np.random.permutation(len(x_train))

    x_train = x_train[perm]
    y_train = y_train[perm]

    total_loss = 0
    correct = 0

    # ==
    # Mini-batch loop
    # ==
    for i in range(0, len(x_train), batchSize):

        batch_end = min(i + batchSize, len(x_train))

        # 
        # Create batch
        # 
        x_batch = x_train[i:batch_end][:, None, :, :] / 255.0
        y_batch = y_train[i:batch_end]

        B = x_batch.shape[0]

        # 
        # Forward
        # 
        probs, cache, runningMean1, runningVar1, runningMean2, runningVar2 = (
            forwardPass(
                x_batch,
                convFilter1,
                convBias1,
                convFilter2,
                convBias2,
                denseWeight,
                denseBias,
                gamma1,
                beta1,
                gamma2,
                beta2,
                runningMean1,
                runningVar1,
                runningMean2,
                runningVar2,
            )
        )
        # 
        # Loss
        # 
        loss = 0

        for k in range(B):

            loss += cross_entropy(probs[:, k : k + 1], y_batch[k])

        loss /= B

        total_loss += loss * B

        # 
        # Accuracy
        # 
        pred = np.argmax(probs, axis=0)
        correct += np.sum(pred == y_batch)

        # 
        # Backward
        # 
        grads = backwardPass(
            probs, y_batch, cache, convFilter1, convFilter2, denseWeight
        )

        (
            dconvFilter1,
            dconvBias1,
            dconvFilter2,
            dconvBias2,
            ddenseWeight,
            ddenseBias,
            dgamma1,
            dbeta1,
            dgamma2,
            dbeta2,
        ) = grads

        # 
        # Adam step
        # 
        t += 1

        #  Conv1 
        convFilter1, mconvFilter1, vconvFilter1 = adamW(
            convFilter1,
            dconvFilter1,
            mconvFilter1,
            vconvFilter1,
            t,
            lr,
            0.9,
            0.999,
            1e-8,
            weight_decay,
        )

        convBias1, mconvBias1, vconvBias1 = adamOptimize(
            convBias1, dconvBias1, mconvBias1, vconvBias1, t, lr, 0.9, 0.999, 1e-8
        )

        #  Conv2 
        convFilter2, mconvFilter2, vconvFilter2 = adamW(
            convFilter2,
            dconvFilter2,
            mconvFilter2,
            vconvFilter2,
            t,
            lr,
            0.9,
            0.999,
            1e-8,
            weight_decay,
        )

        convBias2, mconvBias2, vconvBias2 = adamOptimize(
            convBias2, dconvBias2, mconvBias2, vconvBias2, t, lr, 0.9, 0.999, 1e-8
        )

        #  Dense 
        denseWeight, mdenseWeight, vdenseWeight = adamW(
            denseWeight,
            ddenseWeight,
            mdenseWeight,
            vdenseWeight,
            t,
            lr,
            0.9,
            0.999,
            1e-8,
            weight_decay,
        )

        denseBias, mdenseBias, vdenseBias = adamOptimize(
            denseBias, ddenseBias, mdenseBias, vdenseBias, t, lr, 0.9, 0.999, 1e-8
        )

        #  BatchNorm1 
        gamma1, mg1, vg1 = adamOptimize(
            gamma1, dgamma1, mg1, vg1, t, lr, 0.9, 0.999, 1e-8
        )

        beta1, mb1, vb1 = adamOptimize(beta1, dbeta1, mb1, vb1, t, lr, 0.9, 0.999, 1e-8)

        #  BatchNorm2 
        gamma2, mg2, vg2 = adamOptimize(
            gamma2, dgamma2, mg2, vg2, t, lr, 0.9, 0.999, 1e-8
        )

        beta2, mb2, vb2 = adamOptimize(beta2, dbeta2, mb2, vb2, t, lr, 0.9, 0.999, 1e-8)

        # 
        # Training progress
        # 
        if i % 320 == 0:
            step_time = time.time() - epoch_start
    
            print(
                f"Epoch {epoch+1}/{epochs} "
                f"| Step {i:5d}/{len(x_train)} "
                f"| Batch Loss {loss:.4f} "
                f"| Running Acc {(correct/(i+B))*100:.2f}% "
                f"| Time {step_time:.2f}s"
            )

    # ==
    # Epoch summary
    # ==
    epoch_loss = total_loss / len(x_train)
    epoch_acc = correct / len(x_train)
    epoch_time = time.time() - epoch_start

    print("\n")
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Loss     : {epoch_loss:.4f}")
    print(f"Accuracy : {epoch_acc:.4f}")
    print(f"Time     : {epoch_time:.2f}s")
    print("\n")

    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)
    # ==
    # Save best model
    # ==
    if epoch_loss < bestLoss:

        bestLoss = epoch_loss

        np.savez(
            "MNIST3_model.npz",
            #  Conv 
            convFilter1=convFilter1,
            convBias1=convBias1,
            convFilter2=convFilter2,
            convBias2=convBias2,
            #  Dense 
            denseWeight=denseWeight,
            denseBias=denseBias,
            #  BatchNorm params 
            gamma1=gamma1,
            beta1=beta1,
            gamma2=gamma2,
            beta2=beta2,
            #  Running stats 
            runningMean1=runningMean1,
            runningVar1=runningVar1,
            runningMean2=runningMean2,
            runningVar2=runningVar2,
            #  Adam states Conv1 
            mconvFilter1=mconvFilter1,
            vconvFilter1=vconvFilter1,
            mconvBias1=mconvBias1,
            vconvBias1=vconvBias1,
            #  Adam states Conv2 
            mconvFilter2=mconvFilter2,
            vconvFilter2=vconvFilter2,
            mconvBias2=mconvBias2,
            vconvBias2=vconvBias2,
            #  Adam states Dense 
            mdenseWeight=mdenseWeight,
            vdenseWeight=vdenseWeight,
            mdenseBias=mdenseBias,
            vdenseBias=vdenseBias,
            #  Adam states BN1 
            mg1=mg1,
            vg1=vg1,
            mb1=mb1,
            vb1=vb1,
            #  Adam states BN2 
            mg2=mg2,
            vg2=vg2,
            mb2=mb2,
            vb2=vb2,
            #  Training state 
            t=t,
            bestLoss=bestLoss,
        )


plt.figure()

#  Loss curve 
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

#  Accuracy curve 
plt.subplot(1, 2, 2)
plt.plot(train_accs)
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.tight_layout()
plt.show()
