# Eval.py

import numpy as np
import matplotlib.pyplot as plt

from Models.Restnet9 import ResNet9

from Cifar import x_test, y_test, class_names



from Loss.cross_entropy import (
    CrossEntropyLoss
)

# ==========================================
# LOAD DATA
# ==========================================


print("Test Images :", x_test.shape)
print("Test Labels :", y_test.shape)

# ==========================================
# LOAD MODEL
# ==========================================

model = ResNet9()

model.load(
    "resnet9_cifar.npz"
)

model.eval()

# ==========================================
# SETTINGS
# ==========================================

batch_size = 32

loss_fn = CrossEntropyLoss()

# ==========================================
# METRICS
# ==========================================

correct = 0
total = 0

total_loss = 0

all_preds = []
all_targets = []

# ==========================================
# EVALUATION LOOP
# ==========================================

for i in range(
    0,
    len(x_test),
    batch_size
):

    batch_end = min(
        i + batch_size,
        len(x_test)
    )

    x_batch = x_test[i:batch_end]

    y_batch = y_test[i:batch_end]

    # ----------------------------------
    # FORWARD
    # ----------------------------------

    logits = model.forward(x_batch)

    # ----------------------------------
    # LOSS
    # ----------------------------------

    loss = loss_fn.forward(
        logits,
        y_batch
    )

    total_loss += (
        loss * len(x_batch)
    )

    # ----------------------------------
    # PREDICTIONS
    # ----------------------------------

    preds = np.argmax(
        logits,
        axis=1
    )

    # ----------------------------------
    # STORE RESULTS
    # ----------------------------------

    all_preds.extend(preds)

    all_targets.extend(y_batch)

    # ----------------------------------
    # ACCURACY
    # ----------------------------------

    correct += np.sum(
        preds == y_batch
    )

    total += len(x_batch)

    # ----------------------------------
    # PROGRESS
    # ----------------------------------

    if i % 320 == 0:

        running_acc = (
            correct / total
        ) * 100

        print(
            f"Step {i:5d}/{len(x_test)} "
            f"| Acc {running_acc:.2f}%"
        )

# ==========================================
# FINAL RESULTS
# ==========================================

test_acc = (
    correct / total
) * 100

test_loss = (
    total_loss / total
)

print("\n==========================")
print("FINAL TEST RESULTS")
print("==========================")

print(
    f"Test Loss     : {test_loss:.4f}"
)

print(
    f"Test Accuracy : {test_acc:.2f}%"
)

# ==========================================
# CONFUSION MATRIX
# ==========================================

num_classes = 10

conf_matrix = np.zeros(
    (num_classes, num_classes),
    dtype=np.int32
)

for true, pred in zip(
    all_targets,
    all_preds
):

    conf_matrix[
        true,
        pred
    ] += 1

# ==========================================
# PLOT CONFUSION MATRIX
# ==========================================

plt.figure(figsize=(10, 8))

plt.imshow(conf_matrix)

plt.title("Confusion Matrix")

plt.colorbar()

ticks = np.arange(num_classes)

plt.xticks(
    ticks,
    class_names,
    rotation=45
)

plt.yticks(
    ticks,
    class_names
)

plt.xlabel("Predicted")

plt.ylabel("True")

# ------------------------------------------
# WRITE NUMBERS INSIDE CELLS
# ------------------------------------------

for i in range(num_classes):

    for j in range(num_classes):

        plt.text(
            j,
            i,
            str(conf_matrix[i, j]),
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.show()
