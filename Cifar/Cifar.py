import pickle
import numpy as np


# ==========================================
# LOAD SINGLE BATCH
# ==========================================

def load_cifar_batch(filename):

    with open(filename, "rb") as f:

        batch = pickle.load(
            f,
            encoding="bytes"
        )

    # -------------------------
    # IMAGES
    # -------------------------

    x = batch[b"data"]

    # shape:
    # (10000, 3072)

    x = x.reshape(
        -1,
        3,
        32,
        32
    )

    x = x.astype(np.float32)

    mean = np.array(
        [0.4914, 0.4822, 0.4465],
        dtype=np.float32
    ).reshape(1,3,1,1)

    std = np.array(
        [0.2470, 0.2435, 0.2616],
        dtype=np.float32
    ).reshape(1,3,1,1)

    x = x.astype(np.float32) / 255.0
    x = (x - mean) / std

    # -------------------------
    # LABELS
    # -------------------------

    y = np.array(
        batch[b"labels"]
    )

    return x, y


# ==========================================
# LOAD TRAIN
# ==========================================

def load_cifar10_train(path):

    xs = []
    ys = []

    for i in range(1, 6):

        filename = (
            f"{path}/data_batch_{i}"
        )

        x, y = load_cifar_batch(
            filename
        )

        xs.append(x)
        ys.append(y)

    x_train = np.concatenate(xs)

    y_train = np.concatenate(ys)

    return x_train, y_train


# ==========================================
# LOAD TEST
# ==========================================

def load_cifar10_test(path):

    filename = (
        f"{path}/test_batch"
    )

    return load_cifar_batch(
        filename
    )


# ==========================================
# LOAD LABEL NAMES
# ==========================================

def load_cifar_labels(path):

    with open(
        f"{path}/batches.meta",
        "rb"
    ) as f:

        meta = pickle.load(
            f,
            encoding="bytes"
        )

    labels = [

        label.decode("utf-8")

        for label in meta[
            b"label_names"
        ]
    ]

    return labels

path=r"C:\Users\HI TCC\Code\WD_regconition\cifar-10-batches-py"


x_train, y_train = load_cifar10_train(path)

x_test, y_test = load_cifar10_test(path)
class_names = load_cifar_labels(path)


