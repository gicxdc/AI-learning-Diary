import numpy as np
from Foward import *
from Other import *
import matplotlib.pyplot as plt


# 
# PREDICT


def predict(
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

    if x.ndim == 3:
        x = x[None, :, :, :]  # (1,C,H,W)

    B = x.shape[0]

    #  Conv1 
    conv1_out = []

    for b in range(B):
        out, _ = convForward_im2col(
            x[b],
            convFilter1,
            convBias1,
            stride=1,
            padding=1
        )
        conv1_out.append(out)

    conv1_out = np.stack(conv1_out)

    bn1 = gamma1 * (
        (conv1_out - runningMean1)
        / np.sqrt(runningVar1 + 1e-5)
    ) + beta1

    relu1 = reluFoward(bn1)

    #  Pool1 
    pool1 = []

    for b in range(B):
        out, _ = maxpoolForward_im2col(relu1[b], 2, 2)
        pool1.append(out)

    pool1 = np.stack(pool1)

    #  Conv2 
    conv2_out = []

    for b in range(B):
        out, _ = convForward_im2col(
            pool1[b],
            convFilter2,
            convBias2,
            stride=1,
            padding=1
        )
        conv2_out.append(out)

    conv2_out = np.stack(conv2_out)

    bn2 = gamma2 * (
        (conv2_out - runningMean2)
        / np.sqrt(runningVar2 + 1e-5)
    ) + beta2

    relu2 = reluFoward(bn2)

    #  Pool2 
    pool2 = []

    for b in range(B):
        out, _ = maxpoolForward_im2col(relu2[b], 2, 2)
        pool2.append(out)

    pool2 = np.stack(pool2)

    #  Flatten 
    flat = pool2.reshape(B, -1).T

    #  Dense 
    logits, _ = denseForward(flat, denseWeight, denseBias)

    probs = softmaxFoward(logits)

    preds = np.argmax(probs, axis=0)

    return preds, probs




def load_model(path="MNIST3_model.npz"):

    data = np.load(path)

    return (
        data["convFilter1"],
        data["convBias1"],
        data["convFilter2"],
        data["convBias2"],
        data["denseWeight"],
        data["denseBias"],
        data["gamma1"],
        data["beta1"],
        data["gamma2"],
        data["beta2"],
        data["runningMean1"],
        data["runningVar1"],
        data["runningMean2"],
        data["runningVar2"],
    )




def evaluate_test(
    x_test,
    y_test,
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

    total_loss = 0
    correct = 0

    for i in range(len(x_test)):

        x = x_test[i].reshape(1, 28, 28) / 255.0
        y = y_test[i]

        preds, probs = predict(
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
        )

        loss = cross_entropy(probs[:, 0:1], y)

        total_loss += loss

        if preds[0] == y:
            correct += 1

    return total_loss / len(x_test), correct / len(x_test)



# CONFUSION MATRIX 

def confusion_matrix(
    x_test,
    y_test,
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

    cm = np.zeros((10, 10), dtype=int)

    for i in range(len(x_test)):

        x = x_test[i].reshape(1, 28, 28) / 255.0
        y = y_test[i]

        preds, probs = predict(
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
        )

        pred = preds[0]

        cm[y][pred] += 1

    return cm



# PLOT CONFUSION MATRIX


def plot_confusion_matrix(cm, normalize=False):

    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(8, 6))

    plt.imshow(cm)

    plt.title("Confusion Matrix")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.xticks(np.arange(10))
    plt.yticks(np.arange(10))

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            text = f"{cm[i, j]:.2f}" if normalize else f"{cm[i, j]}"

            plt.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=8
            )

    plt.colorbar()
    plt.tight_layout()
    plt.show()



# PER CLASS ACCURACY


def per_class_accuracy(cm):

    print("\n PER-CLASS ACCURACY ")

    for i in range(10):

        total = np.sum(cm[i])

        correct = cm[i][i]

        acc = correct / total if total > 0 else 0

        print(f"Class {i}: {acc:.4f}")




x_test = read_idx(r"D:\TeraBoxDownload\archive\t10k-images.idx3-ubyte")
y_test = read_idx(r"D:\TeraBoxDownload\archive\t10k-labels.idx1-ubyte")


# LOAD MODEL


(
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
) = load_model("MNIST3_model.npz")


test_loss, test_acc = evaluate_test(
    x_test,
    y_test,
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

print("\n TEST RESULTS ")
print("Loss:", test_loss)
print("Accuracy:", test_acc)


cm = confusion_matrix(
    x_test,
    y_test,
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

plot_confusion_matrix(cm)

per_class_accuracy(cm)