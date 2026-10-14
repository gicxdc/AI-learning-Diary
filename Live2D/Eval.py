import numpy as np
import matplotlib.pyplot as plt
from Foward import *
from Other import *

def forwardPass(x,
                convFilter1, convBias1,
                convFilter2, convBias2,
                denseWeight, denseBias):

    # ===== Conv Block 1 =====
    convOut1, cache1 = convForward_im2col(x, convFilter1, convBias1, stride=1,padding=1)
    reluOut1 = reluFoward(convOut1) 
    poolOut1, poolCache1 = maxpoolForward_im2col(reluOut1, 2, 2)

    # ===== Conv Block 2 =====
    convOut2, cache2 = convForward_im2col(poolOut1, convFilter2, convBias2, stride=1,padding=1)
    reluOut2 = reluFoward(convOut2)
    poolOut2, poolCache2 = maxpoolForward_im2col(reluOut2, 2, 2)

    # ===== Flatten =====
    flat = poolOut2.reshape(-1, 1)
 
    
    # ===== Dense =====
    logits, denseCache = denseForward(flat, denseWeight, denseBias)
    probs = softmaxFoward(logits)

    cache = (x,
             convOut1, reluOut1, poolOut1, poolCache1,
             convOut2, reluOut2, poolOut2, poolCache2,
             flat,
             cache1, cache2,
             denseCache )

    return probs, cache


# =========================
# LOAD MODEL
# =========================
def load_model(path="MNIST_model.npz"):
    data = np.load(path)

    convFilter1 = data["convFilter1"]
    convBias1   = data["convBias1"]

    convFilter2 = data["convFilter2"]
    convBias2   = data["convBias2"]

    denseWeight = data["denseWeight"]
    denseBias   = data["denseBias"]

    # Adam states
    mconvFilter1 = data["mconvFilter1"]
    vconvFilter1 = data["vconvFilter1"]

    mconvFilter2 = data["mconvFilter2"]
    vconvFilter2 = data["vconvFilter2"]

    mdenseWeight = data["mdenseWeight"]
    vdenseWeight = data["vdenseWeight"]

    mconvBias1 = data["mconvBias1"]
    vconvBias1 = data["vconvBias1"]

    mconvBias2 = data["mconvBias2"]
    vconvBias2 = data["vconvBias2"]

    mdenseBias = data["mdenseBias"]
    vdenseBias = data["vdenseBias"]

    t = int(data["t"])
    bestLoss = float(data["bestLoss"])

    return (
        convFilter1, convBias1,
        convFilter2, convBias2,
        denseWeight, denseBias,
    )


# =========================
# TEST EVALUATION
# =========================
def evaluate_test(x_test, y_test, W1, b1, W2, b2, W3, b3):
    total_loss = 0
    correct = 0

    for i in range(len(x_test)):
        x = x_test[i].reshape(1, 28, 28) / 255.0
        y = y_test[i]

        probs, _ = forwardPass(x, W1, b1, W2, b2, W3, b3)

        loss = cross_entropy(probs, y)
        total_loss += loss

        pred = np.argmax(probs)
        if pred == y:
            correct += 1

    return total_loss / len(x_test), correct / len(x_test)


# =========================
# CONFUSION MATRIX
# =========================
def confusion_matrix(x_test, y_test, W1, b1, W2, b2, W3, b3):
    cm = np.zeros((10, 10), dtype=int)

    for i in range(len(x_test)):
        x = x_test[i].reshape(1, 28, 28) / 255.0
        y = y_test[i]

        probs, _ = forwardPass(x, W1, b1, W2, b2, W3, b3)
        pred = np.argmax(probs)

        cm[y][pred] += 1

    return cm


# =========================
# PER CLASS ACCURACY
# =========================
def per_class_accuracy(cm):
    print("\n===== PER-CLASS ACCURACY =====")

    for i in range(10):
        total = np.sum(cm[i])
        correct = cm[i][i]

        acc = correct / total if total > 0 else 0
        print(f"Class {i}: {acc:.4f}")

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

    # numbers inside cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{cm[i, j]:.2f}" if normalize else f"{cm[i, j]}"
            plt.text(j, i, text, ha="center", va="center", fontsize=8)

    plt.colorbar()
    plt.tight_layout()
    plt.show()

# =========================
# MAIN
# =========================

# load dataset
x_test = read_idx(r'D:\TeraBoxDownload\archive\t10k-images.idx3-ubyte')
y_test = read_idx(r'D:\TeraBoxDownload\archive\t10k-labels.idx1-ubyte')

print("Test set:", x_test.shape)

# load trained model
W1, b1, W2, b2, W3, b3 = load_model("MNIST_model.npz")

# evaluate
test_loss, test_acc = evaluate_test(x_test, y_test, W1, b1, W2, b2, W3, b3)

print("\n===== TEST RESULTS =====")
print("Loss:", test_loss)
print("Accuracy:", test_acc)

# confusion matrix
cm = confusion_matrix(x_test, y_test, W1, b1, W2, b2, W3, b3)

plot_confusion_matrix(cm)              # raw counts
plot_confusion_matrix(cm, True) 

per_class_accuracy(cm)

